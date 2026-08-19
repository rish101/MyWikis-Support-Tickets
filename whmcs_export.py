"""
Exports support-ticket data from WHMCS via its API into an Excel file,
for downstream processing and embedding into Pinecone (see pinecone_upload.py).

Pipeline: WHMCS API -> paginated ticket fetch -> per-ticket detail fetch (threaded) -> Excel export

Requires:
    pip install requests pandas openpyxl

Environment:
    WHMCS_API_URL         - your WHMCS API endpoint, e.g. https://panel.example.com/includes/api.php
    WHMCS_API_IDENTIFIER  - your WHMCS API identifier
    WHMCS_API_SECRET      - your WHMCS API secret
"""

import os
import time
import concurrent.futures
import requests
import pandas as pd

# --- Config: loaded from environment, never hardcoded ---
API_URL = os.environ.get("WHMCS_API_URL")
API_IDENTIFIER = os.environ.get("WHMCS_API_IDENTIFIER")
API_SECRET = os.environ.get("WHMCS_API_SECRET")

if not API_URL or not API_IDENTIFIER or not API_SECRET:
    raise EnvironmentError(
        "Set WHMCS_API_URL, WHMCS_API_IDENTIFIER, and WHMCS_API_SECRET environment "
        "variables before running."
    )

OUTPUT_FILE = "support_tickets_full.xlsx"
PAGE_SIZE = 100
MAX_WORKERS = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

# Collected ticket rows, populated by worker threads
ticket_data = []


def get_tickets(limit: int = 100, start: int = 0) -> list[dict]:
    """Fetch a page of ticket summaries from WHMCS."""
    print(f"Fetching tickets starting from {start}")
    payload = {
        "action": "GetTickets",
        "identifier": API_IDENTIFIER,
        "secret": API_SECRET,
        "responsetype": "json",
        "limitnum": limit,
        "limitstart": start,
    }

    try:
        response = requests.post(API_URL, data=payload, headers=HEADERS, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

    if response.status_code == 200:
        response_data = response.json()
        if (
            response_data.get("result") == "success"
            and "tickets" in response_data
            and "ticket" in response_data["tickets"]
        ):
            return response_data["tickets"]["ticket"]
    return []


def get_ticket_details(ticket: dict) -> None:
    """Fetch full details (including messages) for a single ticket and append to ticket_data."""
    ticket_id = ticket.get("id")
    payload = {
        "action": "GetTicket",
        "identifier": API_IDENTIFIER,
        "secret": API_SECRET,
        "ticketid": ticket_id,
        "responsetype": "json",
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(API_URL, data=payload, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                ticket_details = response.json()

                if ticket_details.get("result") == "success":
                    ticket_info = {
                        "Ticket ID": ticket_details.get("ticketid"),
                        "Subject": ticket_details.get("subject", "N/A"),
                        "Status": ticket_details.get("status", "N/A"),
                        "Priority": ticket_details.get("priority", "N/A"),
                        "Client Name": ticket_details.get("name", "N/A"),
                        "Date": ticket_details.get("date", "N/A"),
                        "Last Reply": ticket_details.get("lastreply", "N/A"),
                        "Department Name": ticket_details.get("deptname", "N/A"),
                        "Admin": ticket_details.get("admin", "N/A"),
                        "Service": ticket_details.get("service", "N/A"),
                        "CC Recipients": ticket_details.get("cc", "N/A"),
                    }

                    if "replies" in ticket_details and "reply" in ticket_details["replies"]:
                        replies = ticket_details["replies"]["reply"]
                        messages = [
                            f"{reply.get('date')}: {reply.get('message', 'N/A')} (by {reply.get('name', 'N/A')})"
                            for reply in replies
                        ]
                        ticket_info["Messages"] = "\n".join(messages)
                    else:
                        ticket_info["Messages"] = "No replies"

                    ticket_data.append(ticket_info)
                    return
                else:
                    print(f"Warning: unexpected response structure for Ticket ID {ticket_id}")
            else:
                print(
                    f"Error fetching details for Ticket ID {ticket_id}: "
                    f"Status Code {response.status_code} - {response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(
                f"Request error for Ticket ID {ticket_id} "
                f"(Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}"
            )

        time.sleep(RETRY_DELAY_SECONDS)

    print(f"Failed to fetch details for Ticket ID {ticket_id} after {RETRY_ATTEMPTS} attempts.")


def main() -> None:
    start = 0
    while True:
        tickets = get_tickets(limit=PAGE_SIZE, start=start)
        if not tickets:
            break

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(get_ticket_details, ticket) for ticket in tickets]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        start += PAGE_SIZE

    if not ticket_data:
        print("No ticket data collected. Exiting without exporting to Excel.")
        return

    df = pd.DataFrame(ticket_data)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Tickets exported successfully to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
