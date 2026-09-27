"""Customer-support tools for a fictitious online store.

The tools share one in-memory dataset so they can be chained:
find orders -> read an order -> track it or check/start a return.
Data and "today" are fixed so evaluations are deterministic across runs.
"""

from datetime import date, timedelta

from langchain.tools import tool

TODAY = date(2026, 9, 27)
RETURN_WINDOW_DAYS = 30

CUSTOMERS = {
    "ana@example.com": {"name": "Ana Lopez", "orders": ["ORD-1001", "ORD-1002", "ORD-1004"]},
    "ben@example.com": {"name": "Ben Carter", "orders": ["ORD-1003"]},
}

ORDERS = {
    "ORD-1001": {
        "placed": date(2026, 9, 10),
        "status": "delivered",
        "delivered": date(2026, 9, 14),
        "items": [
            {"sku": "HD-200", "name": "Wireless Headphones", "price": 129.00, "final_sale": False},
            {"sku": "CB-010", "name": "USB-C Cable", "price": 12.50, "final_sale": False},
        ],
        "shipment": {"carrier": "UPS", "tracking": "1Z999AA10123456784", "events": [
            (date(2026, 9, 11), "Shipped from Madrid warehouse"),
            (date(2026, 9, 14), "Delivered, left at front door"),
        ]},
    },
    "ORD-1002": {
        "placed": date(2026, 9, 24),
        "status": "in_transit",
        "delivered": None,
        "items": [
            {"sku": "KB-310", "name": "Mechanical Keyboard", "price": 89.99, "final_sale": False},
        ],
        "shipment": {"carrier": "DHL", "tracking": "JD014600003456789012", "eta": date(2026, 9, 29), "events": [
            (date(2026, 9, 25), "Shipped from Madrid warehouse"),
            (date(2026, 9, 26), "Arrived at Lyon sorting facility"),
        ]},
    },
    "ORD-1003": {
        "placed": date(2026, 7, 30),
        "status": "delivered",
        "delivered": date(2026, 8, 3),
        "items": [
            {"sku": "MN-270", "name": "27-inch Monitor", "price": 249.00, "final_sale": False},
        ],
        "shipment": {"carrier": "UPS", "tracking": "1Z999AA10987654321", "events": [
            (date(2026, 7, 31), "Shipped from Madrid warehouse"),
            (date(2026, 8, 3), "Delivered, signed by B. Carter"),
        ]},
    },
    "ORD-1004": {
        "placed": date(2026, 9, 26),
        "status": "processing",
        "delivered": None,
        "items": [
            {"sku": "GC-050", "name": "Gift Card 50 EUR", "price": 50.00, "final_sale": True},
            {"sku": "MS-120", "name": "Ergonomic Mouse", "price": 39.90, "final_sale": False},
        ],
        "shipment": None,
    },
}

# Returns opened during this process's lifetime, keyed by (order_id, sku).
RETURNS: dict[tuple[str, str], str] = {}


def _find_item(order: dict, sku: str) -> dict | None:
    return next((item for item in order["items"] if item["sku"] == sku.upper()), None)


def _return_decision(order_id: str, sku: str) -> tuple[bool, str]:
    order = ORDERS.get(order_id.upper())
    if order is None:
        return False, f"Order {order_id} does not exist."
    item = _find_item(order, sku)
    if item is None:
        return False, f"Item {sku} is not part of order {order_id}."
    if (order_id.upper(), item["sku"]) in RETURNS:
        return False, f"A return already exists for {item['name']}: {RETURNS[(order_id.upper(), item['sku'])]}."
    if item["final_sale"]:
        return False, f"{item['name']} is a final-sale item and cannot be returned."
    if order["status"] != "delivered":
        return False, f"Order {order_id} has not been delivered yet (status: {order['status']}). Returns open after delivery."
    deadline = order["delivered"] + timedelta(days=RETURN_WINDOW_DAYS)
    if TODAY > deadline:
        return False, f"The {RETURN_WINDOW_DAYS}-day return window for {item['name']} closed on {deadline.isoformat()}."
    return True, f"{item['name']} can be returned until {deadline.isoformat()} for a refund of {item['price']:.2f} EUR."


@tool
def find_customer_orders(email: str) -> str:
    """List the order IDs and statuses for a customer, looked up by email address."""
    customer = CUSTOMERS.get(email.strip().lower())
    if customer is None:
        return f"No customer found with email {email}."
    lines = [f"Orders for {customer['name']}:"]
    for order_id in customer["orders"]:
        order = ORDERS[order_id]
        lines.append(f"- {order_id}: placed {order['placed'].isoformat()}, status {order['status']}")
    return "\n".join(lines)


@tool
def get_order_details(order_id: str) -> str:
    """Return the items (with SKUs), prices, status and dates of an order."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return f"Order {order_id} does not exist."
    total = sum(item["price"] for item in order["items"])
    lines = [f"Order {order_id.upper()} placed {order['placed'].isoformat()}, status {order['status']}."]
    if order["delivered"]:
        lines.append(f"Delivered on {order['delivered'].isoformat()}.")
    for item in order["items"]:
        final = " (final sale)" if item["final_sale"] else ""
        lines.append(f"- {item['sku']} {item['name']}: {item['price']:.2f} EUR{final}")
    lines.append(f"Total: {total:.2f} EUR")
    return "\n".join(lines)


@tool
def track_shipment(order_id: str) -> str:
    """Return the carrier, tracking number, delivery estimate and tracking history of an order."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return f"Order {order_id} does not exist."
    shipment = order["shipment"]
    if shipment is None:
        return f"Order {order_id.upper()} has not shipped yet (status: {order['status']})."
    lines = [f"{shipment['carrier']} tracking number {shipment['tracking']}."]
    if "eta" in shipment:
        lines.append(f"Estimated delivery: {shipment['eta'].isoformat()}.")
    for day, event in shipment["events"]:
        lines.append(f"- {day.isoformat()}: {event}")
    return "\n".join(lines)


@tool
def check_return_eligibility(order_id: str, sku: str) -> str:
    """Check whether an item from an order can be returned, and until when."""
    _, message = _return_decision(order_id.strip(), sku.strip())
    return message


@tool
def create_return(order_id: str, sku: str, reason: str) -> str:
    """Open a return for an eligible item and return the RMA number and next steps."""
    eligible, message = _return_decision(order_id.strip(), sku.strip())
    if not eligible:
        return f"Return not created. {message}"
    key = (order_id.strip().upper(), sku.strip().upper())
    rma = f"RMA-{len(RETURNS) + 5001}"
    RETURNS[key] = rma
    item = _find_item(ORDERS[key[0]], key[1])
    return (
        f"Return {rma} created for {item['name']} (reason: {reason}). "
        f"A prepaid label was sent to the customer's email; the refund of {item['price']:.2f} EUR "
        "is issued within 5 business days of the item arriving at the warehouse."
    )


TOOLS = [
    find_customer_orders,
    get_order_details,
    track_shipment,
    check_return_eligibility,
    create_return,
]
