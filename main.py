import json
import re
import threading
import time
from typing import List, Optional

import anthropic
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tripletex competition bot")


def keep_alive():
    while True:
        time.sleep(600)  # 10 minutes
        try:
            requests.get("https://tripletex-bot.onrender.com/health", timeout=10)
            print("Keep-alive ping sent")
        except Exception:
            pass


threading.Thread(target=keep_alive, daemon=True).start()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class SolveRequest(BaseModel):
    prompt: str
    files: Optional[List[FileAttachment]] = None
    tripletex_credentials: TripletexCredentials


# ---------------------------------------------------------------------------
# Claude: interpret prompt → structured action plan
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an accounting assistant that translates natural-language

Note: Prompts may come in Norwegian. Key Norwegian terms:
- ansatt/medarbeider = employee
- kunde = customer
- leverandør = supplier
- faktura = invoice
- produkt = product
- avdeling = department
- prosjekt = project
- reiseregning = travel expense
- betaling = payment
- kreditnota = credit note

instructions into Tripletex API actions.

Given a user prompt, respond ONLY with a JSON object (no markdown, no explanation).

The JSON must have this shape:
{
  "actions": [
    {
      "type": "<action_type>",
      "payload": { <fields needed for that action> }
    }
  ]
}

Supported action types and their payloads:

create_customer
  payload: { "name": string, "email"?: string, "phoneNumber"?: string,
             "organizationNumber"?: string, "address"?: string,
             "postalCode"?: string, "city"?: string, "country"?: string }

create_supplier
  payload: { "name": string, "email"?: string, "phoneNumber"?: string,
             "organizationNumber"?: string, "address"?: string,
             "postalCode"?: string, "city"?: string, "country"?: string }

create_product
  payload: { "name": string, "number"?: string,
             "priceExcludingVatCurrency"?: number,
             "costExcludingVatCurrency"?: number,
             "vatTypeId"?: number }

Note: NEVER include priceIncludingVatCurrency - it causes validation errors.

create_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "employeeNumber"?: string }

create_invoice
  payload: { "customer_name": string, "invoiceDate": "YYYY-MM-DD",
             "invoiceDueDate": "YYYY-MM-DD",
             "orders": [ { "description": string,
                           "unitPriceExcludingVatCurrency": number,
                           "count": number } ] }

create_project
  payload: { "name": string, "projectManagerEmail"?: string,
             "startDate"?: "YYYY-MM-DD", "endDate"?: "YYYY-MM-DD",
             "customerName"?: string }

create_department
  payload: { "name": string, "departmentNumber"?: string }

create_ledger_posting
  payload: { "description": string, "date": "YYYY-MM-DD",
             "debitAccount": "NNNN", "creditAccount": "NNNN", "amount": number }

create_travel_expense
  payload: { "description": string, "date": "YYYY-MM-DD", "amount": number,
             "category"?: string }

register_payment
  payload: { "customer_name"?: string, "invoiceId"?: number,
             "amount": number, "date": "YYYY-MM-DD" }

create_credit_note
  payload: { "invoiceId": number, "date": "YYYY-MM-DD" }

update_customer
  payload: { "name": string, "email"?: string, "phoneNumber"?: string }

delete_travel_expense
  payload: { "id"?: number, "description"?: string }

update_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "phoneNumber"?: string }

log_hours
  payload: { "employeeEmail": string, "projectName": string, "activityName"?: string,
             "date": "YYYY-MM-DD", "hours": number, "hourlyRate"?: number }

update_product
  payload: { "name": string, "priceExcludingVatCurrency"?: number,
             "priceIncludingVatCurrency"?: number, "costExcludingVatCurrency"?: number }

create_contact
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "phoneNumber"?: string, "customerId"?: number, "customerName"?: string }

delete_customer
  payload: { "name": string }

delete_employee
  payload: { "firstName": string, "lastName": string }

create_product_with_price
  payload: { "name": string, "priceExcludingVatCurrency": number,
             "costExcludingVatCurrency"?: number, "number"?: string }

register_supplier_invoice
  payload: { "supplierName": string, "organizationNumber"?: string,
             "invoiceNumber"?: string, "amount": number, "vatPercent"?: number,
             "accountCode"?: string, "date": "YYYY-MM-DD" }

no_op
  payload: { "reason": string }

Rules:
- If the prompt is ambiguous, do your best and pick the most likely action.
- If you truly cannot map it, use no_op.
- Extract names, dates, amounts from the prompt.
- For dates, use today (2025-03-20) if not specified.
- Return ONLY valid JSON, nothing else.
"""


def ask_claude(prompt: str) -> dict:
    return ask_claude_with_content([{"type": "text", "text": prompt}])


def ask_claude_with_content(content: list) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Tripletex HTTP helpers
# ---------------------------------------------------------------------------

def tx_auth(token: str):
    return ("0", token)


def tx_post(base_url: str, token: str, path: str, payload: dict) -> requests.Response:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    print(f"POST {url}  payload={json.dumps(payload)}")
    r = requests.post(url, auth=tx_auth(token), json=payload, timeout=30)
    print(f"  -> {r.status_code}: {r.text[:300]}")
    return r


def tx_get(base_url: str, token: str, path: str, params: dict = None) -> requests.Response:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    print(f"GET {url}  params={params}")
    r = requests.get(url, auth=tx_auth(token), params=params, timeout=30)
    print(f"  -> {r.status_code}: {r.text[:300]}")
    return r


def set_bank_account(base_url: str, token: str) -> bool:
    """Try multiple endpoints to find and set company bank account."""
    # Try /token/employee first to get company context
    for path in ["/companyWithLoginAccess", "/company/withLoginAccess",
                 "/internaltransaction/company"]:
        r = tx_get(base_url, token, path)
        print(f"Trying {path} -> {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            value = data.get("value", data.get("values", [{}]))
            if isinstance(value, list):
                company = value[0] if value else {}
            else:
                company = value
            company_id = company.get("id")
            if company_id:
                url = f"{base_url.rstrip('/')}/company/{company_id}"
                r2 = requests.put(url, auth=tx_auth(token),
                                 json={"id": company_id,
                                       "bankAccountNumber": "15060126900"},
                                 timeout=30)
                print(f"Set bank account -> {r2.status_code}: {r2.text[:300]}")
                return r2.status_code in (200, 201)
    return False


# ---------------------------------------------------------------------------
# Action executors
# ---------------------------------------------------------------------------

def do_create_customer(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Customer"), "isCustomer": True}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    # Build address if any address fields present
    addr = {}
    if payload.get("address"):
        addr["addressLine1"] = payload["address"]
    if payload.get("postalCode"):
        addr["postalCode"] = payload["postalCode"]
    if payload.get("city"):
        addr["city"] = payload["city"]
    if payload.get("country"):
        addr["country"] = payload["country"]
    if addr:
        body["physicalAddress"] = addr
    tx_post(base_url, token, "/customer", body)


def do_create_supplier(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Supplier"), "isSupplier": True}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    # Build address if any address fields present
    addr = {}
    if payload.get("address"):
        addr["addressLine1"] = payload["address"]
    if payload.get("postalCode"):
        addr["postalCode"] = payload["postalCode"]
    if payload.get("city"):
        addr["city"] = payload["city"]
    if payload.get("country"):
        addr["country"] = payload["country"]
    if addr:
        body["physicalAddress"] = addr
    tx_post(base_url, token, "/customer", body)


def do_create_product(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Product")}
    for field in ("number", "costExcludingVatCurrency", "priceExcludingVatCurrency"):
        if payload.get(field) is not None:
            body[field] = payload[field]
    if payload.get("vatTypeId"):
        body["vatTypeId"] = payload["vatTypeId"]
    r = tx_post(base_url, token, "/product", body)
    if r.status_code == 422 and "er i bruk" in r.text:
        print("Product number in use, retrying without number")
        body.pop("number", None)
        r = tx_post(base_url, token, "/product", body)


def do_update_product(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/product", {"name": name, "count": 1})
    if r.status_code != 200:
        return
    products = r.json().get("values", [])
    if not products:
        print(f"No product found: {name}")
        return
    product_id = products[0]["id"]
    update_body = {}
    for field in ("priceExcludingVatCurrency", "priceIncludingVatCurrency",
                  "costExcludingVatCurrency", "number"):
        if payload.get(field) is not None:
            update_body[field] = payload[field]
    if not update_body:
        return
    url = f"{base_url.rstrip('/')}/product/{product_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /product/{product_id} -> {r2.status_code}: {r2.text[:200]}")


def do_create_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")
    if first and not last:
        parts = first.rsplit(" ", 1)
        first = parts[0]
        last = parts[1] if len(parts) > 1 else "Unknown"
    body = {
        "firstName": first or "Unknown",
        "lastName": last or "Employee",
        "userType": "STANDARD",
    }
    for field in ("email", "employeeNumber"):
        if payload.get(field):
            body[field] = payload[field]
    if not body.get("email"):
        safe_name = (first + "." + last).lower().replace(" ", "")
        body["email"] = f"{safe_name}.{int(time.time())}@example.com"
    dept_r = tx_get(base_url, token, "/department", {"count": 1})
    if dept_r.status_code == 200:
        depts = dept_r.json().get("values", [])
        if depts:
            body["department"] = {"id": depts[0]["id"]}
    r = tx_post(base_url, token, "/employee", body)
    if r.status_code == 422 and "finnes allerede" in r.text:
        print("Employee email exists, looking up existing employee")
        r2 = tx_get(base_url, token, "/employee", {"email": body["email"]})
        if r2.status_code == 200:
            employees = r2.json().get("values", [])
            if employees:
                print(f"Found existing employee id: {employees[0]['id']}")


def do_create_project(base_url: str, token: str, payload: dict) -> None:
    body = {
        "name": payload.get("name", "New Project"),
        "startDate": payload.get("startDate", "2025-03-20"),
        "endDate": payload.get("endDate", "2026-03-20"),
    }
    body["number"] = payload.get("number", "")
    if payload.get("projectManagerEmail"):
        r = tx_get(base_url, token, "/employee",
                   {"email": payload["projectManagerEmail"]})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                body["projectManager"] = {"id": employees[0]["id"]}
    if "projectManager" not in body:
        r = tx_get(base_url, token, "/employee", {"count": 1})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                body["projectManager"] = {"id": employees[0]["id"]}
    tx_post(base_url, token, "/project", body)


def do_create_department(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "New Department")}
    if payload.get("departmentNumber"):
        body["departmentNumber"] = payload["departmentNumber"]
    tx_post(base_url, token, "/department", body)


def do_create_invoice(base_url: str, token: str, payload: dict) -> None:
    customer_name = payload.get("customer_name", "Unknown Customer")
    invoice_date = payload.get("invoiceDate", "2025-03-20")
    due_date = payload.get("invoiceDueDate", "2025-04-20")

    # Step 1: find or create customer
    r = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
    customer_id = None
    if r.status_code == 200:
        customers = r.json().get("values", [])
        if customers:
            customer_id = customers[0]["id"]
    if customer_id is None:
        r2 = tx_post(base_url, token, "/customer",
                     {"name": customer_name, "isCustomer": True})
        if r2.status_code in (200, 201):
            customer_id = r2.json().get("value", {}).get("id")
    if customer_id is None:
        print("Could not find or create customer for invoice - aborting")
        return

    # Step 2: create order
    raw_orders = payload.get("orders") or []
    if not raw_orders:
        raw_orders = [{"description": "Service",
                       "unitPriceExcludingVatCurrency": 0, "count": 1}]
    order_lines = [
        {
            "description": item.get("description", "Item"),
            "unitPriceExcludingVatCurrency": item.get(
                "unitPriceExcludingVatCurrency", 0),
            "count": item.get("count", 1),
        }
        for item in raw_orders
    ]
    order_body = {
        "customer": {"id": customer_id},
        "orderDate": invoice_date,
        "deliveryDate": invoice_date,
        "orderLines": order_lines,
    }
    r3 = tx_post(base_url, token, "/order", order_body)
    if r3.status_code not in (200, 201):
        print(f"Order creation failed ({r3.status_code})")
        return
    order_id = r3.json().get("value", {}).get("id")
    if not order_id:
        print("No order ID returned")
        return

    # Step 3: invoice from order - try different send types
    for send_type in [None, "EMAIL", "EHF", "PAPER"]:
        invoice_body = {
            "invoiceDate": invoice_date,
            "invoiceDueDate": due_date,
            "customer": {"id": customer_id},
            "orders": [{"id": order_id}],
        }
        if send_type:
            invoice_body["sendType"] = send_type
        r_inv = tx_post(base_url, token, "/invoice", invoice_body)
        print(f"Invoice attempt (sendType={send_type}) -> {r_inv.status_code}: {r_inv.text[:200]}")
        if r_inv.status_code in (200, 201):
            print("Invoice created successfully!")
            return
        if "bankkontonummer" not in r_inv.text:
            # Different error - no point retrying with different send type
            break

    print("All invoice attempts failed")


def do_create_ledger_posting(base_url: str, token: str, payload: dict) -> None:
    date = payload.get("date", "2025-03-20")
    description = payload.get("description", "Manual posting")
    amount = payload.get("amount", 0)
    debit = payload.get("debitAccount", "1500")
    credit = payload.get("creditAccount", "4000")

    body = {
        "date": date,
        "description": description,
        "postings": [
            {
                "date": date,
                "description": description,
                "account": {"number": int(debit)},
                "amount": amount,
                "amountCurrency": amount,
            },
            {
                "date": date,
                "description": description,
                "account": {"number": int(credit)},
                "amount": -amount,
                "amountCurrency": -amount,
            }
        ]
    }
    tx_post(base_url, token, "/ledger/voucher", body)


def do_create_travel_expense(base_url: str, token: str, payload: dict) -> None:
    # Find employee to attach expense to
    r_emp = tx_get(base_url, token, "/employee", {"count": 1})
    employee_id = None
    if r_emp.status_code == 200:
        employees = r_emp.json().get("values", [])
        if employees:
            employee_id = employees[0]["id"]
    if not employee_id:
        print("No employee found for travel expense")
        return
    body = {
        "employee": {"id": employee_id},
        "description": payload.get("description", "Travel expense"),
        "travelDetails": {
            "departureDate": payload.get("date", "2025-03-20") + "T08:00:00",
            "returnDate": payload.get("date", "2025-03-20") + "T18:00:00",
            "departureFrom": payload.get("departureFrom", "Oslo"),
            "destination": payload.get("destination", "Bergen"),
            "departureTransportation": "CAR",
            "returnTransportation": "CAR",
        },
        "isCompleted": False,
    }
    tx_post(base_url, token, "/travelExpense", body)


def do_register_payment(base_url: str, token: str, payload: dict) -> None:
    amount = payload.get("amount", 0)
    date = payload.get("date", "2025-03-20")

    # Find the invoice by customer name or directly
    invoice_id = payload.get("invoiceId")
    if not invoice_id:
        customer_name = payload.get("customer_name", "")
        if customer_name:
            r = tx_get(base_url, token, "/customer",
                      {"name": customer_name, "count": 1})
            if r.status_code == 200:
                customers = r.json().get("values", [])
                if customers:
                    customer_id = customers[0]["id"]
                    r2 = tx_get(base_url, token, "/invoice", {
                        "customerId": customer_id,
                        "invoiceDateFrom": "2020-01-01",
                        "invoiceDateTo": "2030-12-31",
                        "count": 1,
                    })
                    if r2.status_code == 200:
                        invoices = r2.json().get("values", [])
                        if invoices:
                            invoice_id = invoices[0]["id"]
        if not invoice_id:
            r = tx_get(base_url, token, "/invoice",
                      {"invoiceDateFrom": "2020-01-01",
                       "invoiceDateTo": "2030-12-31", "count": 1})
            if r.status_code == 200:
                invoices = r.json().get("values", [])
                if invoices:
                    invoice_id = invoices[0]["id"]

    if not invoice_id:
        print("No invoice found to register payment against")
        return

    payment_body = {
        "paymentDate": date,
        "paidAmount": amount,
        "paymentTypeId": 1,
    }
    url = f"{base_url.rstrip('/')}/invoice/{invoice_id}/:payment"
    r2 = requests.put(url, auth=tx_auth(token), json=payment_body, timeout=30)
    print(f"PUT /invoice/{invoice_id}/:payment -> {r2.status_code}: {r2.text[:200]}")


def do_create_credit_note(base_url: str, token: str, payload: dict) -> None:
    invoice_id = payload.get("invoiceId")
    date = payload.get("date", "2025-03-20")
    if not invoice_id:
        print("No invoiceId provided for credit note")
        return
    tx_post(base_url, token, f"/invoice/{invoice_id}/:createCreditNote", {
        "date": date,
    })


def do_update_customer(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/customer", {"name": name, "count": 1})
    if r.status_code != 200:
        print("Could not find customer to update")
        return
    customers = r.json().get("values", [])
    if not customers:
        print(f"No customer found with name: {name}")
        return
    customer_id = customers[0]["id"]
    update_body = {}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            update_body[field] = payload[field]
    if not update_body:
        print("No fields to update")
        return
    url = f"{base_url.rstrip('/')}/customer/{customer_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /customer/{customer_id} -> {r2.status_code}: {r2.text[:200]}")


def do_delete_travel_expense(base_url: str, token: str, payload: dict) -> None:
    expense_id = payload.get("id")
    if not expense_id:
        r = tx_get(base_url, token, "/travelExpense", {"count": 1})
        if r.status_code == 200:
            expenses = r.json().get("values", [])
            if expenses:
                expense_id = expenses[0]["id"]
    if not expense_id:
        print("No travel expense found to delete")
        return
    url = f"{base_url.rstrip('/')}/travelExpense/{expense_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /travelExpense/{expense_id} -> {r2.status_code}")


def do_update_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")
    r = tx_get(base_url, token, "/employee",
               {"firstName": first, "lastName": last, "count": 1})
    if r.status_code != 200:
        return
    employees = r.json().get("values", [])
    if not employees:
        print(f"No employee found: {first} {last}")
        return
    employee_id = employees[0]["id"]
    update_body = {}
    for field in ("email", "phoneNumber"):
        if payload.get(field):
            update_body[field] = payload[field]
    if not update_body:
        return
    url = f"{base_url.rstrip('/')}/employee/{employee_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /employee/{employee_id} -> {r2.status_code}: {r2.text[:200]}")


def do_log_hours(base_url: str, token: str, payload: dict) -> None:
    hours = payload.get("hours", 0)
    date = payload.get("date", "2025-03-20")
    employee_email = payload.get("employeeEmail", "")
    project_name = payload.get("projectName", "")

    # Find employee
    employee_id = None
    if employee_email:
        r = tx_get(base_url, token, "/employee", {"email": employee_email})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                employee_id = employees[0]["id"]

    # Find project
    project_id = None
    if project_name:
        r = tx_get(base_url, token, "/project", {"name": project_name, "count": 1})
        if r.status_code == 200:
            projects = r.json().get("values", [])
            if projects:
                project_id = projects[0]["id"]

    # Find or use default activity
    activity_id = None
    activity_name = payload.get("activityName", "")
    if activity_name:
        r = tx_get(base_url, token, "/activity", {"name": activity_name, "count": 1})
        if r.status_code == 200:
            activities = r.json().get("values", [])
            if activities:
                activity_id = activities[0]["id"]
    if not activity_id:
        r = tx_get(base_url, token, "/activity", {"count": 1})
        if r.status_code == 200:
            activities = r.json().get("values", [])
            if activities:
                activity_id = activities[0]["id"]

    if not employee_id or not project_id or not activity_id:
        print(f"Missing required IDs for timekeeping: employee={employee_id} project={project_id} activity={activity_id}")
        return

    body = {
        "date": date,
        "hours": hours,
        "employee": {"id": employee_id},
        "project": {"id": project_id},
        "activity": {"id": activity_id},
    }
    if payload.get("hourlyRate"):
        body["hourlyRate"] = payload["hourlyRate"]

    tx_post(base_url, token, "/timesheet/entry", body)


def do_create_contact(base_url: str, token: str, payload: dict) -> None:
    body = {
        "firstName": payload.get("firstName", "Unknown"),
        "lastName": payload.get("lastName", "Contact"),
    }
    for field in ("email", "phoneNumber"):
        if payload.get(field):
            body[field] = payload[field]
    if payload.get("customerName"):
        r = tx_get(base_url, token, "/customer",
                   {"name": payload["customerName"], "count": 1})
        if r.status_code == 200:
            customers = r.json().get("values", [])
            if customers:
                body["customer"] = {"id": customers[0]["id"]}
    elif payload.get("customerId"):
        body["customer"] = {"id": payload["customerId"]}
    tx_post(base_url, token, "/contact", body)


def do_delete_customer(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/customer", {"name": name, "count": 1})
    if r.status_code != 200:
        return
    customers = r.json().get("values", [])
    if not customers:
        print(f"No customer found: {name}")
        return
    customer_id = customers[0]["id"]
    url = f"{base_url.rstrip('/')}/customer/{customer_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /customer/{customer_id} -> {r2.status_code}")


def do_delete_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")
    r = tx_get(base_url, token, "/employee",
               {"firstName": first, "lastName": last, "count": 1})
    if r.status_code != 200:
        return
    employees = r.json().get("values", [])
    if not employees:
        print(f"No employee found: {first} {last}")
        return
    employee_id = employees[0]["id"]
    url = f"{base_url.rstrip('/')}/employee/{employee_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /employee/{employee_id} -> {r2.status_code}")


def do_create_product_with_price(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Product")}
    if payload.get("priceExcludingVatCurrency") is not None:
        body["priceExcludingVatCurrency"] = payload["priceExcludingVatCurrency"]
    if payload.get("costExcludingVatCurrency") is not None:
        body["costExcludingVatCurrency"] = payload["costExcludingVatCurrency"]
    if payload.get("number"):
        body["number"] = payload["number"]
    tx_post(base_url, token, "/product", body)


def do_register_supplier_invoice(base_url: str, token: str, payload: dict) -> None:
    supplier_name = payload.get("supplierName", "Unknown Supplier")
    date = payload.get("date", "2025-03-20")
    amount_with_vat = payload.get("amount", 0)
    vat_percent = payload.get("vatPercent", 25)
    account_code = payload.get("accountCode", "6340")
    invoice_number = payload.get("invoiceNumber", "")

    # Calculate amounts
    vat_factor = vat_percent / (100 + vat_percent)
    vat_amount = round(amount_with_vat * vat_factor, 2)
    net_amount = round(amount_with_vat - vat_amount, 2)

    # Find or create supplier
    r = tx_get(base_url, token, "/customer",
               {"name": supplier_name, "count": 1})
    supplier_id = None
    if r.status_code == 200:
        suppliers = r.json().get("values", [])
        if suppliers:
            supplier_id = suppliers[0]["id"]
    if not supplier_id:
        org_no = payload.get("organizationNumber", "")
        body = {"name": supplier_name, "isSupplier": True}
        if org_no:
            body["organizationNumber"] = org_no
        r2 = tx_post(base_url, token, "/customer", body)
        if r2.status_code in (200, 201):
            supplier_id = r2.json().get("value", {}).get("id")

    # Create voucher with postings
    description = f"{invoice_number} - {supplier_name}" if invoice_number else supplier_name
    voucher_body = {
        "date": date,
        "description": description,
        "postings": [
            {
                "date": date,
                "description": description,
                "account": {"number": int(account_code)},
                "amount": net_amount,
                "amountCurrency": net_amount,
            },
            {
                "date": date,
                "description": f"VAT {vat_percent}% - {description}",
                "account": {"number": 2700},
                "amount": vat_amount,
                "amountCurrency": vat_amount,
            },
            {
                "date": date,
                "description": description,
                "account": {"number": 2400},
                "amount": -amount_with_vat,
                "amountCurrency": -amount_with_vat,
            }
        ]
    }
    if supplier_id:
        voucher_body["supplier"] = {"id": supplier_id}
    tx_post(base_url, token, "/ledger/voucher", voucher_body)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "create_customer": do_create_customer,
    "create_supplier": do_create_supplier,
    "create_product": do_create_product,
    "update_product": do_update_product,
    "create_employee": do_create_employee,
    "create_project": do_create_project,
    "create_department": do_create_department,
    "create_invoice": do_create_invoice,
    "create_ledger_posting": do_create_ledger_posting,
    "create_travel_expense": do_create_travel_expense,
    "register_payment": do_register_payment,
    "create_credit_note": do_create_credit_note,
    "update_customer": do_update_customer,
    "delete_travel_expense": do_delete_travel_expense,
    "update_employee": do_update_employee,
    "log_hours": do_log_hours,
    "create_contact": do_create_contact,
    "delete_customer": do_delete_customer,
    "delete_employee": do_delete_employee,
    "create_product_with_price": do_create_product_with_price,
    "register_supplier_invoice": do_register_supplier_invoice,
}


def dispatch(base_url: str, token: str, action: dict) -> None:
    action_type = action.get("type", "no_op")
    payload = action.get("payload", {})
    print(f"Dispatching: {action_type}  payload={payload}")
    if action_type == "no_op":
        print(f"no_op reason: {payload.get('reason', '(none)')}")
        return
    handler = ACTION_MAP.get(action_type)
    if handler:
        try:
            handler(base_url, token, payload)
        except Exception as e:
            print(f"Error in {action_type}: {e}")
    else:
        print(f"Unknown action type: {action_type}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/test-interpret")
def test_interpret(body: dict) -> dict:
    """Debug: returns Claude's action plan for a prompt without calling Tripletex."""
    prompt = body.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    try:
        plan = ask_claude(prompt)
        return {"prompt": prompt, "plan": plan}
    except Exception as e:
        return {"error": str(e)}


@app.post("/solve")
def solve(body: SolveRequest) -> dict:
    print(f"=== incoming prompt: {body.prompt!r} ===")
    base_url = body.tripletex_credentials.base_url
    token = body.tripletex_credentials.session_token
    try:
        user_content = []
        for f in (body.files or []):
            try:
                if f.mime_type == "application/pdf":
                    user_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": f.content_base64,
                        }
                    })
                elif f.mime_type.startswith("image/"):
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f.mime_type,
                            "data": f.content_base64,
                        }
                    })
            except Exception as e:
                print(f"Error processing file {f.filename}: {e}")
        user_content.append({"type": "text", "text": body.prompt})
        plan = ask_claude_with_content(user_content)
        print(f"Claude plan: {json.dumps(plan)}")
    except Exception as e:
        print(f"Claude error: {e} - falling back to no_op")
        return {"status": "completed"}
    for action in plan.get("actions", []):
        dispatch(base_url, token, action)
    return {"status": "completed"}
