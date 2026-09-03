"""
Generates a synthetic support-ticket corpus for local development and testing.

The real WHMCS ticket export is company data and is not in this repo. This script
writes a small, realistic stand-in corpus with the same schema that whmcs_export.py
produces, so the retrieval and generation pipeline can be run end to end by anyone
who clones the repo.

Output: data/synthetic_tickets.json
"""

import json
import os

# Each entry mirrors a row from whmcs_export.py: ticket metadata plus the
# flattened reply thread in "Messages".
TICKETS = [
    {
        "subject": "VisualEditor not loading on my wiki",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "VisualEditor stopped working this morning. Clicking Edit just shows the source editor and the browser console has a 404 for the Parsoid endpoint.",
            "Thanks for reporting. Parsoid was unreachable because the internal REST endpoint URL in your wiki config still pointed at the old app server after last week's migration. I have updated $wgVirtualRestConfig to the new host and restarted the job runner. VisualEditor should load now, please hard refresh.",
            "Confirmed working, thank you.",
        ],
    },
    {
        "subject": "How do I increase the maximum file upload size?",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "We need to upload PDFs larger than 10 MB but the wiki rejects them.",
            "The upload ceiling is set by PHP rather than MediaWiki itself. On your plan we can raise upload_max_filesize and post_max_size to 64 MB. I have applied that and also raised $wgMaxUploadSize to match. Note that files over about 100 MB will still time out on slower connections.",
        ],
    },
    {
        "subject": "SSL certificate expired warning on custom domain",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "Visitors are seeing a certificate expired warning on our custom domain wiki.example.org.",
            "Your Let's Encrypt renewal failed because the domain's CNAME was changed and the HTTP-01 challenge could not resolve back to us. I have reissued the certificate and it is valid again. To avoid this, keep the CNAME pointed at our endpoint or tell us before changing DNS so we can switch you to DNS-01 validation.",
        ],
    },
    {
        "subject": "Request to install the Cargo extension",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "Could you install Cargo on our wiki? We want to build structured tables from templates.",
            "Cargo is installed and enabled. It is on our supported extension list so it will be kept up to date with your MediaWiki version automatically. You will need to declare table definitions in your templates and then run the Cargo recreate-data job from Special:CargoTables before queries return anything.",
        ],
    },
    {
        "subject": "Spam accounts creating pages every night",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "We are getting dozens of spam pages created overnight by new accounts.",
            "I have enabled ConfirmEdit with reCAPTCHA on account creation and on edits that add external links, and turned on AbuseFilter with our baseline spam ruleset. I also blocked the four IP ranges the accounts came from. If spam continues, we can require email confirmation before editing.",
            "Much quieter this morning. Can we also require email confirmation?",
            "Done, $wgEmailConfirmToEdit is now true.",
        ],
    },
    {
        "subject": "Need a full XML dump of our wiki content",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "We need an export of all pages including history for an internal archive.",
            "I have generated a full dumpBackup XML including revision history and placed it in your account's file area as a gzipped archive. It covers the main, Template and Category namespaces. Images are not included in an XML dump, so I have also attached a separate tarball of the uploads directory.",
        ],
    },
    {
        "subject": "Can we set up single sign-on with our company Google accounts?",
        "dept": "Sales",
        "priority": "Medium",
        "messages": [
            "Is SSO available? We would like staff to log in with their work Google accounts.",
            "Yes. We support SAML and OpenID Connect through PluggableAuth. For Google Workspace the usual path is OpenID Connect. SSO is available on the Business plan and above. Once you confirm the upgrade we will need your client ID, client secret and the domain you want restricted to.",
        ],
    },
    {
        "subject": "Wiki is very slow to load pages",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "Pages are taking eight to ten seconds to load since yesterday.",
            "The slowdown came from an expensive Semantic MediaWiki query on your main page that was scanning every page in the wiki on each render. I have added a result limit and enabled query result caching. Load times are back to under a second. Consider using a cached query or a Cargo table for that listing.",
        ],
    },
    {
        "subject": "Restore a page that was deleted by mistake",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "An admin deleted our Onboarding page this morning. Can it be restored?",
            "Deleted pages are recoverable from Special:Undelete by any administrator on your wiki, and the full history is intact. I have restored it for you this time. Your admins can do this themselves in future from Special:Undelete without opening a ticket.",
        ],
    },
    {
        "subject": "Email notifications are not being delivered",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "Watchlist notification emails are not arriving for anyone on our wiki.",
            "Your wiki was sending from a custom address on your own domain, and that domain's SPF record did not include our sending servers, so the mail was being rejected. Either add our SPF include to your DNS or let us send from our default noreply address. I have temporarily switched you to the default so notifications work now.",
        ],
    },
    {
        "subject": "Migrating an existing wiki from another host",
        "dept": "Sales",
        "priority": "Medium",
        "messages": [
            "We have a self-hosted MediaWiki 1.35 install we want to move to you. What do you need?",
            "We need an XML dump with full history, a tarball of the images directory, and a list of installed extensions and any LocalSettings customizations. We will import it into a staging wiki on a current MediaWiki version, upgrade the schema, and let you review before we point DNS. Migration is included at no charge on annual plans.",
        ],
    },
    {
        "subject": "How do I change the wiki skin and add our logo?",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "We want to use Vector 2022 and put our company logo in the corner.",
            "Vector 2022 is available and I have set it as the default skin. For the logo, upload the image to the wiki and then set $wgLogos. I can point that at your uploaded file, or you can edit MediaWiki:Common.css if you want finer control. Send me the filename you want used.",
        ],
    },
    {
        "subject": "Database error on save: Lock wait timeout exceeded",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "Users are getting a database error when saving edits to large template pages.",
            "The errors came from a long-running Cargo data recreation job holding table locks while users were editing the same templates. I have moved that job to run off-peak and increased the innodb_lock_wait_timeout for the job runner. Saves are working normally now.",
        ],
    },
    {
        "subject": "Add a new administrator to our wiki",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "Please make user Jordan an administrator.",
            "Existing bureaucrats on your wiki can do this directly from Special:UserRights without contacting us. I have granted Jordan the sysop group this time. If you have lost bureaucrat access entirely, we can restore it from our side.",
        ],
    },
    {
        "subject": "Question about backup frequency and retention",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "How often are our wikis backed up and how long do you keep backups?",
            "Wikis are backed up nightly with database and uploads captured together. We retain daily backups for 30 days and monthly snapshots for 12 months. Restores from any retained point can be requested through a ticket and typically complete within a few hours.",
        ],
    },
    {
        "subject": "Broken images after we renamed files in bulk",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "We used a bot to rename about 200 files and now many pages show broken image links.",
            "The file redirects were created but the page HTML was still cached with the old names. I ran a cache purge across the affected pages and rebuilt the image links table with refreshLinks. The images are rendering again. For future bulk renames, leaving redirects enabled avoids this entirely.",
        ],
    },
    {
        "subject": "Enable Semantic MediaWiki on our plan",
        "dept": "Sales",
        "priority": "Medium",
        "messages": [
            "Is Semantic MediaWiki something you support and what does it cost?",
            "Yes, SMW is supported and included on Business and Enterprise plans. It is resource intensive so it is not offered on the Starter plan. If you upgrade we will install it, run the initial data rebuild, and set up the job scheduling for you.",
        ],
    },
    {
        "subject": "Invoice question, charged twice this month",
        "dept": "Billing",
        "priority": "Medium",
        "messages": [
            "We appear to have been charged twice for the same billing period.",
            "You were charged once for the renewal and once for a plan upgrade that was applied mid-cycle. The upgrade charge was prorated but the invoice line was unclear. I have refunded the difference and updated the invoice description so it is clearer going forward.",
        ],
    },
    {
        "subject": "Cancel our account at the end of the term",
        "dept": "Billing",
        "priority": "Low",
        "messages": [
            "We will not be renewing. How do we cancel and get our data?",
            "I have set the account to not auto renew, so it stays active through the end of your paid term. Before that date, request a final export and we will provide the XML dump and image tarball. We keep data for 30 days after expiry and then delete it permanently.",
        ],
    },
    {
        "subject": "Restrict wiki reading to logged-in users only",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "We need our wiki to be private, only visible to our staff accounts.",
            "I have set the read permission so anonymous users cannot view pages, and left account creation restricted to administrators. Note that this hides content from search engines but images served from the uploads path can still be reachable by direct URL unless we also enable protected file access, which I have turned on.",
        ],
    },
    {
        "subject": "Upgrade our MediaWiki version to the latest LTS",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "What is involved in moving to the current LTS release?",
            "We handle upgrades for you. The usual process is a staging clone on the target version, extension compatibility checks, a schema update run, and then a short maintenance window to switch over. Most wikis are done in under an hour. Custom extensions not on our supported list may need review first.",
        ],
    },
    {
        "subject": "Search results are stale and missing new pages",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "Newly created pages are not showing up in the wiki search.",
            "The CirrusSearch index had fallen behind because the job queue was backed up behind a large import. I cleared the backlog and forced a reindex of the affected namespaces. New pages are searchable again, usually within a minute of being saved.",
        ],
    },
    {
        "subject": "Can we run a bot account against the API?",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "We want to script bulk page updates using pywikibot.",
            "Yes. Create a dedicated account, grant it the bot group from Special:UserRights, and generate a bot password from Special:BotPasswords with only the grants you need. Please keep write requests to a few per second so the job queue stays healthy. Let us know if you plan a very large run and we will watch the load.",
        ],
    },
    {
        "subject": "Custom domain not resolving after DNS change",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "We pointed our domain at you yesterday but it still does not load.",
            "The CNAME is correct but there was also an old A record for the same hostname, so resolvers were alternating between the two. Remove the A record and keep only the CNAME. Once that propagates the wiki will load and the certificate will issue automatically.",
        ],
    },
    {
        "subject": "Two-factor authentication for admin accounts",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "Can we require 2FA for administrators?",
            "OATHAuth is available and I have enabled it. Administrators can enrol from Special:OATH with any TOTP app. If you want it mandatory rather than optional for the sysop group we can enforce that as well, just confirm and I will apply it.",
        ],
    },
    {
        "subject": "Page history shows edits from an IP we do not recognize",
        "dept": "Technical Support",
        "priority": "High",
        "messages": [
            "There are edits in our history from an unfamiliar IP address. Were we compromised?",
            "Those edits came from an account whose password appeared in a public credential breach, not from a flaw on our side. I have blocked the account, reverted the edits, and invalidated its sessions. I would recommend enabling 2FA for privileged accounts and asking users to reset passwords.",
        ],
    },
    {
        "subject": "Template rendering breaks after extension update",
        "dept": "Technical Support",
        "priority": "Medium",
        "messages": [
            "Since the last update, several infobox templates render with raw wikitext showing.",
            "The update changed how ParserFunctions handles unclosed tags that your templates relied on. I have fixed the two templates that were missing a closing brace and purged the cache. The rendering is correct now. The underlying wikitext was always slightly malformed, the older version was just more forgiving.",
        ],
    },
    {
        "subject": "How many users can we have on the Starter plan?",
        "dept": "Sales",
        "priority": "Low",
        "messages": [
            "Is there a user limit on Starter?",
            "Starter has no hard cap on registered accounts. The practical limits are storage and concurrent traffic rather than user count. Most teams move up to Business for the extension selection and SSO rather than for user limits.",
        ],
    },
    {
        "subject": "Enable subpages in the main namespace",
        "dept": "Technical Support",
        "priority": "Low",
        "messages": [
            "We want to use slash subpages for our documentation in the main namespace.",
            "Subpages are off by default in the main namespace. I have enabled $wgNamespacesWithSubpages for NS_MAIN, so breadcrumb links will now appear on pages with a slash in the title. Existing pages with slashes will start showing parent links immediately.",
        ],
    },
    {
        "subject": "Set up a staging copy of our wiki for testing",
        "dept": "Sales",
        "priority": "Medium",
        "messages": [
            "We would like a sandbox copy of our production wiki to test template changes.",
            "Staging wikis are available on Business and Enterprise plans. We clone production content into a separate wiki with search engine indexing disabled, and can refresh the clone from production on request. Changes on staging do not sync back automatically, they need to be reapplied.",
        ],
    },
]

CLIENTS = [
    "Ada Okafor", "Marcus Lin", "Priya Raman", "Tomas Vidal", "Hannah Brecht",
    "Yusuf Demir", "Elena Sokolova", "Daniel Osei", "Mei Tanaka", "Rafael Costa",
]


def build_rows() -> list[dict]:
    """Build ticket rows matching the schema whmcs_export.py writes."""
    rows = []
    for i, t in enumerate(TICKETS):
        thread = []
        for j, msg in enumerate(t["messages"]):
            author = CLIENTS[i % len(CLIENTS)] if j % 2 == 0 else "Support Staff"
            day = (i % 28) + 1
            thread.append(f"2025-0{(i % 9) + 1}-{day:02d}: {msg} (by {author})")

        rows.append(
            {
                "Ticket ID": 1000 + i,
                "Subject": t["subject"],
                "Status": "Closed",
                "Priority": t["priority"],
                "Client Name": CLIENTS[i % len(CLIENTS)],
                "Date": f"2025-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
                "Department Name": t["dept"],
                "Messages": "\n".join(thread),
            }
        )
    return rows


def main() -> None:
    os.makedirs("data", exist_ok=True)
    rows = build_rows()
    out_path = os.path.join("data", "synthetic_tickets.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} synthetic tickets to {out_path}")


if __name__ == "__main__":
    main()
