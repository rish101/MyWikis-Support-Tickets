import requests
import pandas as pd
import concurrent.futures
import time

# Set WHMCS API details
api_url = 'https://panel.mywikis.com/includes/api.php'
api_identifier = 'vuvHyL3OGpsFa0tizyUnhCk7Efs9DMkb' 
api_secret = 'lEKhLZKezFALi1t1gFEgaIVS1WfliX5V'
  
# Prepare a list for storing ticket details
ticket_data = []

def get_tickets(limit=100, start=0):
    print(f"Fetching tickets starting from {start}")
    payload = {
        'action': 'GetTickets',
        'identifier': api_identifier,
        'secret': api_secret,
        'responsetype': 'json',
        'limitnum': limit,
        'limitstart': start
    }

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(api_url, data=payload, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

    if response.status_code == 200:
        response_data = response.json()
        if response_data.get('result') == 'success' and 'tickets' in response_data and 'ticket' in response_data['tickets']:
            return response_data['tickets']['ticket']
    return []

def get_ticket_details(ticket):
    ticket_id = ticket.get('id')
    payload = {
        'action': 'GetTicket',
        'identifier': api_identifier,
        'secret': api_secret,
        'ticketid': ticket_id,
        'responsetype': 'json',
    }

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # Retry mechanism with extended delay
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(api_url, data=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                ticket_details = response.json()

                # Print full response for analysis
                print(f"Ticket ID: {ticket_id} - Full Response: {ticket_details}")

                # Extract information directly from the response
                if ticket_details.get('result') == 'success':
                    ticket_info = {
                        'Ticket ID': ticket_details.get('ticketid'),
                        'Subject': ticket_details.get('subject', 'N/A'),
                        'Status': ticket_details.get('status', 'N/A'),
                        'Priority': ticket_details.get('priority', 'N/A'),
                        'Client Name': ticket_details.get('name', 'N/A'),
                        'Date': ticket_details.get('date', 'N/A'),
                        'Last Reply': ticket_details.get('lastreply', 'N/A'),
                        'Department Name': ticket_details.get('deptname', 'N/A'),
                        'Admin': ticket_details.get('admin', 'N/A'),
                        'Service': ticket_details.get('service', 'N/A'),
                        'CC Recipients': ticket_details.get('cc', 'N/A')
                    }

                    # Extract replies if they exist
                    if 'replies' in ticket_details and 'reply' in ticket_details['replies']:
                        replies = ticket_details['replies']['reply']
                        messages = []
                        for reply in replies:
                            messages.append(f"{reply.get('date')}: {reply.get('message', 'N/A')} (by {reply.get('name', 'N/A')})")

                        ticket_info['Messages'] = '\n'.join(messages)
                    else:
                        ticket_info['Messages'] = 'No replies'

                    print(f"Adding Ticket ID {ticket_id} to data.")  # Debug to ensure data is being added
                    ticket_data.append(ticket_info)
                    return
                else:
                    print(f"Warning: Unexpected response structure for Ticket ID {ticket_id}")
            else:
                print(f"Error fetching details for Ticket ID {ticket_id}: Status Code {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request error for Ticket ID {ticket_id} (Attempt {attempt + 1}/{retries}): {e}")

        # Longer delay between retries to handle rate limiting
        time.sleep(5)

    # If all retries fail
    print(f"Failed to fetch details for Ticket ID {ticket_id} after {retries} attempts.")
    return None

# Fetch all tickets with pagination (100 per batch)
start = 0
limit = 100
while True:
    tickets = get_tickets(limit=limit, start=start)
    if not tickets:
        break

    # Use threading to fetch ticket details faster (limit number of threads for efficiency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_ticket_details, ticket) for ticket in tickets]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # The result is handled within the function

    start += limit

# Check if ticket_data has been populated
if not ticket_data:
    print("No ticket data collected. Exiting without exporting to Excel.")
else:
    # Convert the ticket details list to a DataFrame
    df = pd.DataFrame(ticket_data)

    # Export the DataFrame to an Excel file
    output_file = 'support_tickets_full.xlsx'
    df.to_excel(output_file, index=False)

    print(f"Tickets exported successfully to {output_file}")
