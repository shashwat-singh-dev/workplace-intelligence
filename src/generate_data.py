import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from bisect import bisect_left
RUN_VALIDATION = False

# Seeding random values
np.random.seed(42)

print("Workplace data generator started!")

# ====================================================================================================================================
# Generate Employees Dataset
# =====================================================================================================================================

num_employees = 1000

departments = [
    "AI",
    "Engineering",
    "HR",
    "Finance",
    "Sales",
    "Marketing",
    "Operations"
]

work_modes = [
    "Hybrid",
    "Office",
    "Remote"
]

zones = [
    "Quiet",
    "Collaborative",
    "Open"
]

floors = [1, 2, 3, 4, 5]

employees = pd.DataFrame({
    "employee_id": [
        f"E{i:04d}" for i in range(1, num_employees + 1)
    ],

    "department": np.random.choice(
        departments,
        num_employees
    ),

    "work_mode": np.random.choice(
        work_modes,
        num_employees,
        p=[0.60, 0.25, 0.15]
    ),

    "preferred_floor": np.random.choice(
        floors,
        num_employees
    ),

    "preferred_zone": np.random.choice(
        zones,
        num_employees,
        p=[0.30, 0.30, 0.40]
    ),

    "preferred_start_hour": np.random.choice(
        [8, 9, 10],
        num_employees,
        p=[0.20, 0.55, 0.25]
    ),

    # --------------------------------------------
    # Personal Room Booking Preferences
    # --------------------------------------------

    "preferred_room_type": np.random.choice(
        [
            "Meeting",
            "Conference",
            "Training",
            "Focus"
        ],
        num_employees,
        p=[
            0.45,
            0.30,
            0.15,
            0.10
        ]
    ),

    "typical_capacity": np.random.choice(
        [
            2,
            4,
            6,
            8,
            10,
            12,
            16
        ],
        num_employees,
        p=[
            0.15,
            0.25,
            0.15,
            0.20,
            0.10,
            0.10,
            0.05
        ]
    ),

    "typical_duration_minutes": np.random.choice(
        [
            30,
            45,
            60,
            75,
            90,
            105,
            120
        ],
        num_employees,
        p=[
            0.12,
            0.12,
            0.30,
            0.17,
            0.20,
            0.04,
            0.05
        ]
    ),

    "preferred_flexibility_minutes": np.random.choice(
        [
            0,
            15,
            30,
            60,
            120
        ],
        num_employees,
        p=[
            0.20,
            0.20,
            0.25,
            0.25,
            0.10
        ]
    )
})

print("\nEmployees Dataset:")
print(employees.head())

print("\nEmployee dataset shape:")
print(employees.shape)
print("\nEmployee Room Preferences:")
print(
    employees[
        [
            "employee_id",
            "preferred_room_type",
            "typical_capacity",
            "typical_duration_minutes",
            "preferred_flexibility_minutes"
        ]
    ].head(10)
)

# Converted and saved as csv file "employees.csv"
employees.to_csv(
    "data/raw/employees.csv",
    index=False
)

print("Employees dataset saved!")





# ========================================================================================================================================
# Generate Meeting Rooms Dataset
# ========================================================================================================================================

num_rooms = 50

room_types = [
    "Meeting",
    "Conference",
    "Focus",
    "Training"
]

room_sizes = [
    "Small",
    "Medium",
    "Large"
]

# Exactly 10 rooms per floor
room_floors = np.repeat(
    floors,
    num_rooms // len(floors)
)

rooms = pd.DataFrame({
    "room_id": [f"R{i:03d}" for i in range(1, num_rooms + 1)],

    "floor": room_floors,

    "room_type": np.random.choice(
        room_types,
        num_rooms
    ),

    "room_size": np.random.choice(
        room_sizes,
        num_rooms,
        p=[0.40, 0.40, 0.20]
    )
})

# Assign capacity based on room size
rooms["capacity"] = rooms["room_size"].map({
    "Small": 4,
    "Medium": 8,
    "Large": 16
})

print("\nMeeting Rooms Dataset:")
print(rooms.head())

print("\nRoom dataset shape:")
print(rooms.shape)

print("\nRooms per floor:")
print(rooms["floor"].value_counts().sort_index())

# Save it as csv file
rooms.to_csv(
    "data/raw/rooms.csv",
    index=False
)

print("Rooms dataset saved!")

print("\n" + "=" * 60)
print("ROOM INVENTORY ANALYSIS")
print("=" * 60)

print("\nTotal rooms:")
print(len(rooms))

print("\nRooms by type:")
print(
    rooms["room_type"]
    .value_counts()
    .sort_index()
)

print("\nRooms by capacity:")
print(
    rooms["capacity"]
    .value_counts()
    .sort_index()
)

print("\nRooms by floor:")
print(
    rooms["floor"]
    .value_counts()
    .sort_index()
)

print("\nRoom Type × Capacity:")
print(
    rooms.groupby(
        ["room_type", "capacity"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\nRoom Type Summary:")
print(
    rooms.groupby("room_type")["capacity"]
    .agg(["count", "min", "max", "mean", "median"])
)

# ============================================================
# ROOM CAPACITY INDEXING
# ============================================================

room_capacity_index = {}

for _, room in rooms.iterrows():

    room_type = room["room_type"]
    floor = room["floor"]
    capacity = room["capacity"]
    room_id = room["room_id"]

    key = (room_type, floor)

    if key not in room_capacity_index:
        room_capacity_index[key] = []

    room_capacity_index[key].append(
        (room_id, capacity)
    )


print("\nRoom Capacity Index:")

for key, room_list in room_capacity_index.items():
    print(key, "->", room_list)

# ============================================================================================================
# ROOM BOOKING REQUEST GENERATION
# =============================================================================================================

num_room_bookings = 40000

# Business dates: Monday to Friday
business_dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31",
    freq="B"
)

# Higher workplace demand on Monday,
# lower demand on Friday
weekday_weights = {
    0: 1.20,   # Monday
    1: 1.10,   # Tuesday
    2: 1.05,   # Wednesday
    3: 1.10,   # Thursday
    4: 0.80    # Friday
}

date_weights = np.array([
    weekday_weights[d.weekday()]
    for d in business_dates
])

date_weights = date_weights / date_weights.sum()


# --------------------------------------------
# Select employees for booking requests
# --------------------------------------------

work_mode_weights = employees["work_mode"].map({
    "Office": 1.00,
    "Hybrid": 0.55,
    "Remote": 0.12
}).values

work_mode_weights = (
    work_mode_weights / work_mode_weights.sum()
)

employee_indices = np.random.choice(
    employees.index,
    size=num_room_bookings,
    p=work_mode_weights
)

booking_employees = employees.loc[
    employee_indices
].reset_index(drop=True)


# --------------------------------------------
# Select booking dates
# --------------------------------------------

booking_dates = np.random.choice(
    business_dates,
    size=num_room_bookings,
    p=date_weights
)

# --------------------------------------------
# Employee-wise preference consistency
# --------------------------------------------

employee_preference_strength = {}

for employee_id in booking_employees["employee_id"].unique():

    strength = np.random.beta(
        18,
        3
    )

    employee_preference_strength[employee_id] = strength

# --------------------------------------------
# Requested room types
# --------------------------------------------

requested_room_types = []

for employee_id, preferred_type in zip(
    booking_employees["employee_id"],
    booking_employees["preferred_room_type"]
):

    preference_strength = (
        employee_preference_strength[employee_id]
    )

    follow_preference = (
        np.random.random()
        < preference_strength
    )

    if follow_preference:

        requested_type = preferred_type

    else:

        other_types = [
            room_type
            for room_type in [
                "Meeting",
                "Conference",
                "Training",
                "Focus"
            ]
            if room_type != preferred_type
        ]

        requested_type = np.random.choice(
            other_types
        )

    requested_room_types.append(
        requested_type
    )


requested_room_types = np.array(
    requested_room_types
)

# --------------------------------------------
# Employee-wise floor preference consistency
# --------------------------------------------

employee_floor_preference_strength = {}

for employee_id in booking_employees["employee_id"].unique():

    strength = np.random.beta(
        18,
        3
    )

    employee_floor_preference_strength[employee_id] = strength


# --------------------------------------------
# Requested floor
# --------------------------------------------

requested_floors = []

for employee_id, preferred_floor in zip(
    booking_employees["employee_id"],
    booking_employees["preferred_floor"]
):

    preference_strength = (
        employee_floor_preference_strength[employee_id]
    )

    follow_preference = (
        np.random.random()
        < preference_strength
    )

    if follow_preference:

        requested_floor = preferred_floor

    else:

        other_floors = [
            floor
            for floor in floors
            if floor != preferred_floor
        ]

        floor_weights = np.array([
            1 / abs(floor - preferred_floor)
            for floor in other_floors
        ])

        floor_weights = (
            floor_weights /
            floor_weights.sum()
        )

        requested_floor = np.random.choice(
            other_floors,
            p=floor_weights
        )

    requested_floors.append(
        requested_floor
    )


requested_floors = np.array(
    requested_floors
)


# --------------------------------------------
# Requested capacity
# --------------------------------------------

requested_capacity = []

capacity_options = [
    2,
    4,
    6,
    8,
    10,
    12,
    16
]

for typical_capacity in booking_employees["typical_capacity"]:

    # Start with zero probability for every capacity
    probabilities = {
        capacity: 0.0
        for capacity in capacity_options
    }

    # Strong preference for the employee's
    # typical capacity
    probabilities[typical_capacity] = 0.50

    # Nearby capacities are also common
    nearby_capacities = [
        capacity
        for capacity in capacity_options
        if abs(capacity - typical_capacity) <= 2
        and capacity != typical_capacity
    ]

    for capacity in nearby_capacities:
        probabilities[capacity] = 0.20

    # Slight chance of a capacity further away
    other_capacities = [
        capacity
        for capacity in capacity_options
        if probabilities[capacity] == 0.0
    ]

    for capacity in other_capacities:
        probabilities[capacity] = 0.025

    probability_array = np.array([
        probabilities[capacity]
        for capacity in capacity_options
    ])

    probability_array = (
        probability_array
        / probability_array.sum()
    )

    requested_capacity.append(
        np.random.choice(
            capacity_options,
            p=probability_array
        )
    )


requested_capacity = np.array(
    requested_capacity
)


# --------------------------------------------
# Requested start time
# --------------------------------------------

requested_start_hours = []

for hour in booking_employees["preferred_start_hour"]:

    offset = np.random.choice(
        [-1, 0, 0, 0, 0, 1]
    )

    requested_hour = np.clip(
        hour + offset,
        8,
        17
    )

    requested_start_hours.append(
        requested_hour
    )


requested_start_hours = np.array(
    requested_start_hours
)


requested_start_minutes = np.random.choice(
    [0, 15, 30, 45],
    size=num_room_bookings,
    p=[0.25, 0.35, 0.25, 0.15]
)


# --------------------------------------------
# Requested duration
# --------------------------------------------

duration_category = np.random.choice(
    [
        "30-60",
        "60-90",
        "90-120"
    ],
    size=num_room_bookings,
    p=[
        0.35,
        0.50,
        0.15
    ]
)

requested_duration_minutes = []

for category in duration_category:

    if category == "30-60":
        duration = np.random.choice(
            [30, 45, 60]
        )

    elif category == "60-90":
        duration = np.random.choice(
            [60, 75, 90]
        )

    else:
        duration = np.random.choice(
            [90, 105, 120]
        )

    requested_duration_minutes.append(
        duration
)


requested_duration_minutes = np.array(
    requested_duration_minutes
)

# --------------------------------------------
# Maximum time flexibility
# --------------------------------------------

max_time_flexibility_minutes = np.random.choice(
    [
        0,
        15,
        30,
        60,
        120
    ],
    size=num_room_bookings,
    p=[
        0.20,
        0.20,
        0.25,
        0.25,
        0.10
    ]
)


# --------------------------------------------
# Create datetime
# --------------------------------------------

requested_start_datetime = []

for date, hour, minute in zip(
    booking_dates,
    requested_start_hours,
    requested_start_minutes
):

    requested_start_datetime.append(
        pd.Timestamp(date)
        + pd.Timedelta(
            hours=int(hour),
            minutes=int(minute)
        )
    )


requested_start_datetime = pd.to_datetime(
    requested_start_datetime
)


requested_end_datetime = (
    requested_start_datetime
    + pd.to_timedelta(
        requested_duration_minutes,
        unit="m"
    )
)


# --------------------------------------------
# Create request dataset
# --------------------------------------------

room_booking_requests = pd.DataFrame({

    "booking_id": [
        f"RB{i:05d}"
        for i in range(1, num_room_bookings + 1)
    ],

    "employee_id":
        booking_employees["employee_id"].values,

    "booking_date":
        pd.to_datetime(booking_dates).date,

    "requested_start_datetime":
        requested_start_datetime,

    "requested_end_datetime":
        requested_end_datetime,

    "requested_duration_minutes":
        requested_duration_minutes,

    "requested_room_type":
        requested_room_types,

    "requested_capacity":
        requested_capacity,

    "max_time_flexibility_minutes":
        max_time_flexibility_minutes,

    "requested_floor":
        requested_floors
})

# --------------------------------------------
# VERIFY ROOM TYPE PREFERENCE
# --------------------------------------------

room_type_check = room_booking_requests[
    ["employee_id", "requested_room_type"]
].merge(
    employees[
        ["employee_id", "preferred_room_type"]
    ],
    on="employee_id",
    how="left"
)

room_type_check["follows_preference"] = (
    room_type_check["requested_room_type"]
    == room_type_check["preferred_room_type"]
)

print("\nOverall room type preference adherence:")
print(
    room_type_check["follows_preference"].mean()
)

print("\nEmployee-wise room type adherence:")
print(
    room_type_check
    .groupby("employee_id")["follows_preference"]
    .mean()
    .describe()
)

# --------------------------------------------
# VERIFY CAPACITY PREFERENCE
# --------------------------------------------

capacity_check = room_booking_requests[
    ["employee_id", "requested_capacity"]
].merge(
    employees[
        ["employee_id", "typical_capacity"]
    ],
    on="employee_id",
    how="left"
)

capacity_check["matches_typical_capacity"] = (
    capacity_check["requested_capacity"]
    == capacity_check["typical_capacity"]
)

print("\nOverall typical capacity match:")
print(
    capacity_check["matches_typical_capacity"].mean()
)

# --------------------------------------------
# VERIFY CAPACITY NEARNESS
# --------------------------------------------

capacity_check["capacity_difference"] = (
    capacity_check["requested_capacity"]
    - capacity_check["typical_capacity"]
).abs()

print("\nCapacity difference distribution:")
print(
    capacity_check["capacity_difference"]
    .value_counts()
    .sort_index()
)

print("\nCapacity nearness percentages:")

print(
    "Exact match:",
    (
        capacity_check["capacity_difference"] == 0
    ).mean()
)

print(
    "Within ±2:",
    (
        capacity_check["capacity_difference"] <= 2
    ).mean()
)

print(
    "Within ±4:",
    (
        capacity_check["capacity_difference"] <= 4
    ).mean()
)

# --------------------------------------------
# VERIFY FLOOR PREFERENCE
# --------------------------------------------

floor_check = room_booking_requests[
    ["employee_id", "requested_floor"]
].merge(
    employees[
        ["employee_id", "preferred_floor"]
    ],
    on="employee_id",
    how="left"
)

floor_check["floor_difference"] = (
    floor_check["requested_floor"]
    - floor_check["preferred_floor"]
).abs()

print("\nFloor difference distribution:")
print(
    floor_check["floor_difference"]
    .value_counts()
    .sort_index()
)

print("\nFloor preference percentages:")

print(
    "Exact floor:",
    (
        floor_check["floor_difference"] == 0
    ).mean()
)

print(
    "Within ±1 floor:",
    (
        floor_check["floor_difference"] <= 1
    ).mean()
)

print(
    "Within ±2 floors:",
    (
        floor_check["floor_difference"] <= 2
    ).mean()
)

# --------------------------------------------
# VERIFY REQUEST DATASET
# --------------------------------------------

print(room_booking_requests.columns)

print(
    room_booking_requests[
        [
            "booking_id",
            "requested_start_datetime",
            "max_time_flexibility_minutes"
        ]
    ].head(10)
)


# --------------------------------------------
# Sort requests chronologically
# --------------------------------------------

room_booking_requests = room_booking_requests.sort_values(
    [
        "booking_date",
        "requested_start_datetime"
    ]
).reset_index(drop=True)


# --------------------------------------------
# Display results
# --------------------------------------------

print("\nRoom Booking Requests:")
print(room_booking_requests.head())

print("\nRoom Booking Request Shape:")
print(room_booking_requests.shape)

print("\nRequested Room Types:")
print(
    room_booking_requests[
        "requested_room_type"
    ].value_counts()
)

print("\nRequested Capacities:")
print(
    room_booking_requests[
        "requested_capacity"
    ].value_counts().sort_index()
)

print("\nRequested Floors:")
print(
    room_booking_requests[
        "requested_floor"
    ].value_counts().sort_index()
)


# --------------------------------------------
# Save raw booking requests
# --------------------------------------------

room_booking_requests.to_csv(
    "data/raw/room_booking_requests.csv",
    index=False
)

print("\nRoom booking requests saved!")

# ============================================================
# ROOM BOOKING REQUEST ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ROOM BOOKING REQUEST ANALYSIS")
print("=" * 60)

print("\nTotal requests:")
print(len(room_booking_requests))


# ------------------------------------------------------------
# 1. ROOM TYPE
# ------------------------------------------------------------

print("\nRoom Type Distribution:")
print(
    room_booking_requests["requested_room_type"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 2. CAPACITY
# ------------------------------------------------------------

print("\nRequested Capacity Distribution:")
print(
    room_booking_requests["requested_capacity"]
    .value_counts()
    .sort_index()
)

print("\nCapacity Statistics:")
print(
    room_booking_requests["requested_capacity"].describe()
)


# ------------------------------------------------------------
# 3. START HOUR
# ------------------------------------------------------------

print("\nStart Hour Distribution:")
print(
    room_booking_requests["requested_start_datetime"]
    .dt.hour
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 4. DURATION
# ------------------------------------------------------------

print("\nDuration Distribution:")
print(
    room_booking_requests["requested_duration_minutes"]
    .value_counts()
    .sort_index()
)

print("\nDuration Statistics:")
print(
    room_booking_requests["requested_duration_minutes"].describe()
)


# ------------------------------------------------------------
# 5. TIME FLEXIBILITY
# ------------------------------------------------------------

print("\nTime Flexibility Distribution:")
print(
    room_booking_requests["max_time_flexibility_minutes"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 6. REQUESTED FLOOR
# ------------------------------------------------------------

print("\nRequested Floor Distribution:")
print(
    room_booking_requests["requested_floor"]
    .value_counts()
    .sort_index()
)

# ============================================================
# VERIFY BOOKING DATE COVERAGE AND WEEKDAY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("BOOKING DATE AND WEEKDAY VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Verify date range
# ------------------------------------------------------------

print("\nMinimum booking date:")
print(room_booking_requests["booking_date"].min())

print("\nMaximum booking date:")
print(room_booking_requests["booking_date"].max())

print("\nUnique booking dates:")
print(room_booking_requests["booking_date"].nunique())


# ------------------------------------------------------------
# 2. Booking count by weekday
# ------------------------------------------------------------

weekday_analysis = (
    room_booking_requests
    .assign(
        weekday=room_booking_requests[
            "requested_start_datetime"
        ].dt.day_name()
    )
    .groupby("weekday")
    .size()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday"
    ])
)

print("\nBooking requests by weekday:")
print(weekday_analysis)


# ------------------------------------------------------------
# 3. Percentage distribution
# ------------------------------------------------------------

weekday_percentages = (
    weekday_analysis
    / weekday_analysis.sum()
    * 100
)

print("\nBooking percentage by weekday:")
print(weekday_percentages.round(2))


# ------------------------------------------------------------
# 4. Verify ordering
# ------------------------------------------------------------

print("\nHighest-demand weekday:")
print(weekday_analysis.idxmax())

print("\nLowest-demand weekday:")
print(weekday_analysis.idxmin())


# ================================================================================================================================
# ROOM ALLOCATION ENGINE
# ================================================================================================================================

print("\nStarting room allocation engine...")

# ------------------------------------------------------------
# Similar room type hierarchy
# ------------------------------------------------------------

similar_room_types = {
    "Conference": ["Meeting", "Training", "Focus"],
    "Meeting": ["Conference", "Training", "Focus"],
    "Training": ["Conference", "Meeting", "Focus"],
    "Focus": ["Meeting", "Conference", "Training"]
}


# ------------------------------------------------------------
# Prepare room information
# ------------------------------------------------------------

room_records = rooms.to_dict("records")

room_lookup = {
    room["room_id"]: room
    for room in room_records
}

room_floor_lookup = rooms.set_index(
    "room_id"
)["floor"].to_dict()

room_type_lookup = rooms.set_index(
    "room_id"
)["room_type"].to_dict()

room_capacity_lookup = rooms.set_index(
    "room_id"
)["capacity"].to_dict()


# ------------------------------------------------------------
# Store completed room bookings for collision checking
# ------------------------------------------------------------

occupied_rooms = {
    room["room_id"]: []
    for room in room_records
}

room_last_end = {}


# ------------------------------------------------------------
# Helper function: check room availability
# ------------------------------------------------------------

# ============================================================
# INTERVAL-JUMPING AVAILABILITY CHECK
# ============================================================

def get_next_available_time(
    room_id,
    requested_start,
    requested_duration,
    room_bookings
):

    candidate_start = pd.Timestamp(requested_start)

    duration = pd.Timedelta(
        minutes=int(requested_duration)
    )

    existing_bookings = room_bookings.get(
        room_id,
        []
    )

    last_end = room_last_end.get(room_id)

    if last_end is None or candidate_start >= last_end:
        return candidate_start

#=============================== BISECTION / BINARY SEARCH ======================================================================================

    index = bisect_left(
    existing_bookings,
    (candidate_start,)
)
    if index > 0:
        previous_start, previous_end = existing_bookings[index - 1]

        if candidate_start < previous_end:
            candidate_start = previous_end
            index -= 1

    while index < len(existing_bookings):

        existing_start, existing_end = existing_bookings[index]

        if candidate_start + duration <= existing_start:
            break

        if candidate_start >= existing_end:
            index += 1
            continue

        candidate_start = existing_end
        index += 1

    return candidate_start

# ============================================================
# TEST INTERVAL JUMPING
# ============================================================

test_room_bookings = {

    "R021": [
        (
            pd.Timestamp("2025-03-10 09:00"),
            pd.Timestamp("2025-03-10 10:30")
        ),
        (
            pd.Timestamp("2025-03-10 11:00"),
            pd.Timestamp("2025-03-10 12:00")
        )
    ]
}


test_time = get_next_available_time(
    room_id="R021",
    requested_start="2025-03-10 10:00",
    requested_duration=60,
    room_bookings=test_room_bookings
)

print("\nRequested time:")
print("10:00")

print("\nNext available time for R021:")
print(test_time)


# ------------------------------------------------------------
# Helper function: get floors by distance
# ------------------------------------------------------------

def get_floor_order(requested_floor):

    floors_by_distance = sorted(
        floors,
        key=lambda floor: (
            abs(floor - requested_floor),
            floor
        )
    )

    return floors_by_distance


# ------------------------------------------------------------
# Helper function: search rooms
# ------------------------------------------------------------

def find_room(
    start_datetime,
    duration,
    requested_type,
    requested_floor,
    requested_capacity
):

    floor_order = get_floor_order(
        requested_floor
    )

    # --------------------------------------------------------
    # Helper: check candidate rooms
    # --------------------------------------------------------

    def check_candidates(candidate_rooms, allocation_rule):

        for room_id, room_capacity in candidate_rooms:

            # Capacity is a HARD constraint
            if room_capacity < requested_capacity:
                continue

            # Ask interval-jumping function:
            # "When does this room become free?"
            next_available = get_next_available_time(
                room_id,
                start_datetime,
                duration,
                occupied_rooms
            )

            # Room is available at requested time
            if next_available == start_datetime:

                room = room_lookup[room_id]

                return (
                    room,
                    allocation_rule,
                    next_available
                )

        return None


    # ========================================================
    # STEP 1
    # Exact Type + Exact Floor
    # ========================================================

    candidates = room_capacity_index.get(
        (requested_type, requested_floor),
        []
    )

    result = check_candidates(
        candidates,
        "exact_type_exact_floor"
    )

    if result is not None:
        return result


    # ========================================================
    # STEP 2
    # Exact Type + Nearby Floor
    # ========================================================

    nearby_floors = [
        floor
        for floor in floor_order
        if (
            floor != requested_floor
            and abs(floor - requested_floor) == 1
        )
    ]

    for floor in nearby_floors:

        candidates = room_capacity_index.get(
            (requested_type, floor),
            []
        )

        result = check_candidates(
            candidates,
            "exact_type_nearby_floor"
        )

        if result is not None:
            return result


    # ========================================================
    # STEP 3
    # Exact Type + Any Floor
    # ========================================================

    for floor in floor_order:

        candidates = room_capacity_index.get(
            (requested_type, floor),
            []
        )

        result = check_candidates(
            candidates,
            "exact_type_any_floor"
        )

        if result is not None:
            return result


    # ========================================================
    # STEP 4
    # Similar Type + Exact Floor
    # ========================================================

    similar_types = similar_room_types[
        requested_type
    ]

    for similar_type in similar_types:

        candidates = room_capacity_index.get(
            (similar_type, requested_floor),
            []
        )

        result = check_candidates(
            candidates,
            "similar_type_exact_floor"
        )

        if result is not None:
            return result


    # ========================================================
    # STEP 5
    # Similar Type + Any Floor
    # ========================================================

    for similar_type in similar_types:

        for floor in floor_order:

            candidates = room_capacity_index.get(
                (similar_type, floor),
                []
            )

            result = check_candidates(
                candidates,
                "similar_type_any_floor"
            )

            if result is not None:
                return result


    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    return None, "no_room_available", None





# ------------------------------------------------------------
# Allocate every room booking request
# ------------------------------------------------------------

room_booking_records = []

skipped_count = 0
start_time = time.perf_counter()
for i, (_, request) in enumerate(
    room_booking_requests.iterrows()
):

    requested_start = pd.Timestamp(
        request["requested_start_datetime"]
    )

    duration = int(
        request["requested_duration_minutes"]
    )

    max_time_flexibility = int(
    request["max_time_flexibility_minutes"]
    )

    requested_end = (
        requested_start
        + pd.Timedelta(minutes=duration)
    )

    allocated_room = None
    allocation_rule = None
    allocated_start = None
    allocated_end = None
    time_relaxation = None


    # --------------------------------------------------------
    # Try each time relaxation level
    # --------------------------------------------------------

    for relaxation in range(
    0,
    max_time_flexibility + 1,
    15
    ):

        if relaxation == 0:

            possible_times = [
                requested_start
            ]

        else:

            possible_times = [
                requested_start
                - pd.Timedelta(minutes=relaxation),

                requested_start
                + pd.Timedelta(minutes=relaxation)
            ]


        # ----------------------------------------------------
        # Try each candidate time
        # ----------------------------------------------------

        for candidate_start in possible_times:

            candidate_end = (
                candidate_start
                + pd.Timedelta(minutes=duration)
            )


            

            find_room_start = time.perf_counter()

            room, rule, available_time = find_room(
                candidate_start,
                duration,
                request["requested_room_type"],
                int(request["requested_floor"]),
                int(request["requested_capacity"])
            )

            find_room_time = time.perf_counter() - find_room_start

            if i % 500 == 0:
                print(
                    f"Request {i}: "
                    f"find_room took {find_room_time:.4f} seconds"
                )


            if room is not None:

                allocated_room = room
                allocation_rule = rule
                allocated_start = candidate_start
                allocated_end = candidate_end
                time_relaxation = relaxation

                break


        if allocated_room is not None:
            break


    # --------------------------------------------------------
    # Successful allocation
    # --------------------------------------------------------

    if allocated_room is not None:

        room_id = allocated_room["room_id"]

        booking = (
            allocated_start,
            allocated_end
        )

        insert_position = bisect_left(
            occupied_rooms[room_id],
            booking
        )

        occupied_rooms[room_id].insert(
            insert_position,
            booking
        )

        room_last_end[room_id] = max(
        room_last_end.get(room_id, allocated_end),
        allocated_end
        )

        room_booking_records.append({

            "booking_id":
                request["booking_id"],

            "employee_id":
                request["employee_id"],

            "booking_date":
                request["booking_date"],

            "requested_start_datetime":
                requested_start,

            "allocated_start_datetime":
                allocated_start,

            "requested_end_datetime":
                requested_end,

            "allocated_end_datetime":
                allocated_end,

            "requested_duration_minutes":
                duration,

            "max_time_flexibility_minutes":
                max_time_flexibility,

            "requested_room_type":
                request["requested_room_type"],

            "allocated_room_id":
                room_id,

            "allocated_room_type":
                allocated_room["room_type"],

            "requested_floor":
                request["requested_floor"],

            "allocated_floor":
                allocated_room["floor"],

            "requested_capacity":
                request["requested_capacity"],

            "room_capacity":
                allocated_room["capacity"],

            "time_relaxation_minutes":
                time_relaxation,

            "allocation_rule":
                allocation_rule,

            "booking_status":
                "Allocated"
        })


    # --------------------------------------------------------
    # No room found
    # --------------------------------------------------------

    else:

        skipped_count += 1

        room_booking_records.append({

            "booking_id":
                request["booking_id"],

            "employee_id":
                request["employee_id"],

            "booking_date":
                request["booking_date"],

            "requested_start_datetime":
                requested_start,

            "allocated_start_datetime":
                pd.NaT,

            "requested_end_datetime":
                requested_end,

            "allocated_end_datetime":
                pd.NaT,

            "requested_duration_minutes":
                duration,

            "max_time_flexibility_minutes":
                max_time_flexibility,

            "requested_room_type":
                request["requested_room_type"],

            "allocated_room_id":
                None,

            "allocated_room_type":
                None,

            "requested_floor":
                request["requested_floor"],

            "allocated_floor":
                None,

            "requested_capacity":
                request["requested_capacity"],

            "room_capacity":
                None,

            "time_relaxation_minutes":
                None,

            "allocation_rule":
                "skipped_no_room_available",

            "booking_status":
                "Skipped"
        })


# ------------------------------------------------------------
# Create final room booking dataset
# ------------------------------------------------------------

room_bookings = pd.DataFrame(
    room_booking_records
)


# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

room_bookings = room_bookings.sort_values(
    [
        "booking_date",
        "allocated_start_datetime",
        "booking_id"
    ],
    na_position="last"
).reset_index(drop=True)


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

room_bookings.to_csv(
    "data/raw/room_bookings.csv",
    index=False
)

# ============================================================
# SKIPPED REQUEST ANALYSIS
# ============================================================

skipped_requests = pd.read_csv(
    "data/raw/room_bookings.csv",
    parse_dates=[
        "requested_start_datetime",
        "requested_end_datetime"
    ]
)

skipped_requests = skipped_requests[
    skipped_requests["booking_status"] == "Skipped"
].copy()

print("\n" + "=" * 60)
print("SKIPPED REQUEST ANALYSIS")
print("=" * 60)

print("\nTotal skipped:")
print(len(skipped_requests))


print("\nSkipped by Room Type:")
print(
    skipped_requests["requested_room_type"]
    .value_counts()
    .sort_index()
)


print("\nSkipped by Capacity:")
print(
    skipped_requests["requested_capacity"]
    .value_counts()
    .sort_index()
)


print("\nSkipped by Start Hour:")
print(
    skipped_requests["requested_start_datetime"]
    .dt.hour
    .value_counts()
    .sort_index()
)


print("\nSkipped by Duration:")
print(
    skipped_requests["requested_duration_minutes"]
    .value_counts()
    .sort_index()
)


print("\nSkipped by Flexibility:")
print(
    skipped_requests["max_time_flexibility_minutes"]
    .value_counts()
    .sort_index()
)


print("\nSkipped by Requested Floor:")
print(
    skipped_requests["requested_floor"]
    .value_counts()
    .sort_index()
)

print("\nSample Skipped Requests:")

print(
    skipped_requests[
        [
            "booking_id",
            "employee_id",
            "requested_room_type",
            "requested_floor",
            "requested_capacity",
            "requested_start_datetime",
            "requested_end_datetime",
            "requested_duration_minutes",
            "max_time_flexibility_minutes"
        ]
    ].head(30).to_string(index=False)
)




if RUN_VALIDATION:

    # ============================================================
    # ALLOCATION ENGINE VALIDATION
    # ============================================================

    # overlap check
    # capacity check
    # time flexibility check
    # allocation rule check
    # hierarchy tests
    # ============================================================
    # ALLOCATION ENGINE VALIDATION
    # Verify that the allocation engine satisfies all constraints
    # ============================================================

    # ------------------------------------------------------------
    # 1. Check for overlapping bookings within the same room
    # ------------------------------------------------------------

    allocated_bookings = room_bookings[
        room_bookings["booking_status"] == "Allocated"
    ].copy()

    allocated_bookings["allocated_start_datetime"] = pd.to_datetime(
        allocated_bookings["allocated_start_datetime"]
    )

    allocated_bookings["allocated_end_datetime"] = pd.to_datetime(
        allocated_bookings["allocated_end_datetime"]
    )


    allocated_bookings = allocated_bookings.sort_values(
        ["allocated_room_id", "allocated_start_datetime"]
    )

    overlap_count = 0

    for room_id, room_group in allocated_bookings.groupby("allocated_room_id"):

        previous_end = None

        for _, booking in room_group.iterrows():

            if (
                previous_end is not None
                and booking["allocated_start_datetime"] < previous_end
            ):
                overlap_count += 1

            previous_end = max(
                previous_end,
                booking["allocated_end_datetime"]
            ) if previous_end is not None else booking["allocated_end_datetime"]

    print(f"Room overlap violations: {overlap_count}")

    # ------------------------------------------------------------
    # 2. Check room capacity constraint
    # ------------------------------------------------------------

    capacity_violations = (
        allocated_bookings["room_capacity"]
        < allocated_bookings["requested_capacity"]
    ).sum()

    print(f"Room capacity violations: {capacity_violations}")

    # ------------------------------------------------------------
    # 3. Check time-flexibility constraint
    # ------------------------------------------------------------

    time_flexibility_violations = (
        allocated_bookings["time_relaxation_minutes"]
        > allocated_bookings["max_time_flexibility_minutes"]
    ).sum()

    print(
        f"Time flexibility violations: "
        f"{time_flexibility_violations}"
    )

    # ------------------------------------------------------------
    # 4. Validate allocation-rule consistency
    # ------------------------------------------------------------

    rule_violations = 0

    for _, booking in allocated_bookings.iterrows():

        rule = booking["allocation_rule"]

        requested_type = booking["requested_room_type"]
        allocated_type = booking["allocated_room_type"]

        requested_floor = booking["requested_floor"]
        allocated_floor = booking["allocated_floor"]

        if rule == "exact_type_exact_floor":
            valid = (
                allocated_type == requested_type
                and allocated_floor == requested_floor
            )

        elif rule in [
            "exact_type_nearby_floor",
            "exact_type_any_floor"
        ]:
            valid = allocated_type == requested_type

        elif rule == "similar_type_exact_floor":
            valid = (
                allocated_type != requested_type
                and allocated_floor == requested_floor
            )

        elif rule == "similar_type_any_floor":
            valid = allocated_type != requested_type

        else:
            valid = False

        if not valid:
            rule_violations += 1

    print(f"Allocation rule violations: {rule_violations}")

    # ------------------------------------------------------------
    # 5. Test allocation hierarchy with controlled scenarios
    # ------------------------------------------------------------

    print("\nAllocation Hierarchy Tests:")

    original_occupied_rooms = occupied_rooms

    hierarchy_test_results = []

    test_start = pd.Timestamp("2025-12-01 09:00")
    test_duration = 60
    test_capacity = 8


    def run_hierarchy_test(
        test_name,
        requested_type,
        requested_floor,
        target_rule,
        target_room_id
    ):

        global occupied_rooms

        # Make every room unavailable except the target room
        occupied_rooms = {
            room_id: []
            for room_id in room_lookup
        }

        # Block every room except the intended target
        for room_id in occupied_rooms:

            if room_id != target_room_id:

                occupied_rooms[room_id] = [
                    (
                        test_start,
                        test_start + pd.Timedelta(minutes=120)
                    )
                ]

        room, rule, _ = find_room(
            test_start,
            test_duration,
            requested_type,
            requested_floor,
            test_capacity
        )

        passed = (
            room is not None
            and room["room_id"] == target_room_id
            and rule == target_rule
        )

        hierarchy_test_results.append(
            (
                test_name,
                "PASS" if passed else "FAIL"
            )
        )


    # Find suitable rooms for each hierarchy level

    test_rooms = rooms[
        rooms["capacity"] >= test_capacity
    ].copy()


    # Rule 1: Exact type + exact floor
    rule1_room = test_rooms.iloc[0]

    run_hierarchy_test(
        "Rule 1 - Exact type + exact floor",
        rule1_room["room_type"],
        rule1_room["floor"],
        "exact_type_exact_floor",
        rule1_room["room_id"]
    )


    # Rule 2: Exact type + nearby floor
    rule2_room = None

    for _, room in test_rooms.iterrows():

        nearby = test_rooms[
            (test_rooms["room_type"] == room["room_type"])
            &
            (abs(test_rooms["floor"] - room["floor"]) == 1)
        ]

        if len(nearby) > 0:
            rule2_room = nearby.iloc[0]
            requested_type = room["room_type"]
            requested_floor = room["floor"]
            break

    if rule2_room is not None:

        run_hierarchy_test(
            "Rule 2 - Exact type + nearby floor",
            requested_type,
            requested_floor,
            "exact_type_nearby_floor",
            rule2_room["room_id"]
        )


    # Rule 3: Exact type + any floor
    rule3_room = None

    for _, room in test_rooms.iterrows():

        candidates = test_rooms[
            (test_rooms["room_type"] == room["room_type"])
            &
            (abs(test_rooms["floor"] - room["floor"]) > 1)
        ]

        if len(candidates) > 0:
            rule3_room = candidates.iloc[0]
            requested_type = room["room_type"]
            requested_floor = room["floor"]
            break

    if rule3_room is not None:

        run_hierarchy_test(
            "Rule 3 - Exact type + any floor",
            requested_type,
            requested_floor,
            "exact_type_any_floor",
            rule3_room["room_id"]
        )


    # Rule 4: Similar type + exact floor
    rule4_room = None

    for _, room in test_rooms.iterrows():

        similar_types = similar_room_types[room["room_type"]]

        candidates = test_rooms[
            (test_rooms["room_type"].isin(similar_types))
            &
            (test_rooms["floor"] == room["floor"])
        ]

        if len(candidates) > 0:
            rule4_room = candidates.iloc[0]
            requested_type = room["room_type"]
            requested_floor = room["floor"]
            break

    if rule4_room is not None:

        run_hierarchy_test(
            "Rule 4 - Similar type + exact floor",
            requested_type,
            requested_floor,
            "similar_type_exact_floor",
            rule4_room["room_id"]
        )


    # Rule 5: Similar type + any floor
    rule5_room = None

    for _, room in test_rooms.iterrows():

        similar_types = similar_room_types[room["room_type"]]

        candidates = test_rooms[
            (test_rooms["room_type"].isin(similar_types))
            &
            (test_rooms["floor"] != room["floor"])
        ]

        if len(candidates) > 0:
            rule5_room = candidates.iloc[0]
            requested_type = room["room_type"]
            requested_floor = room["floor"]
            break

    if rule5_room is not None:

        run_hierarchy_test(
            "Rule 5 - Similar type + any floor",
            requested_type,
            requested_floor,
            "similar_type_any_floor",
            rule5_room["room_id"]
        )


    # Restore actual booking state
    occupied_rooms = original_occupied_rooms


    for test_name, result in hierarchy_test_results:
        print(f"{test_name}: {result}")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

end_time = time.perf_counter()
print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

print("\nRoom Booking Dataset:")
print(room_bookings.head())

print("\nRoom Booking Dataset Shape:")
print(room_bookings.shape)

print("\nBooking Status:")
print(
    room_bookings["booking_status"].value_counts()
)

print("\nAllocation Rules:")
print(
    room_bookings[
        "allocation_rule"
    ].value_counts()
)

print("\nTime Relaxation:")
print(
    room_bookings[
        "time_relaxation_minutes"
    ].value_counts(dropna=False).sort_index()
)

print("\nSkipped Bookings:")
print(skipped_count)
print("\nRoom bookings saved!")

print("\n" + "=" * 60)
print("ROOM BOOKINGS ANALYSIS")
print("=" * 60)

print("\nTotal bookings:")
print(len(room_bookings))

print("\nBooking Status:")
print(
    room_bookings["booking_status"]
    .value_counts()
)

print("\nAllocation Rule:")
print(
    room_bookings["allocation_rule"]
    .value_counts()
)

print("\nAllocated Room Type:")
print(
    room_bookings["allocated_room_type"]
    .value_counts()
)

print("\nAllocated Capacity:")
print(
    room_bookings["room_capacity"]
    .value_counts()
    .sort_index()
)

print("\nTime Relaxation Used:")
print(
    room_bookings["time_relaxation_minutes"]
    .value_counts()
    .sort_index()
)


employee_behavior = (
    room_booking_requests
    .groupby("employee_id")
    .agg(
        total_requests=("booking_id", "count"),
        unique_room_types=("requested_room_type", "nunique"),
        unique_capacities=("requested_capacity", "nunique"),
        unique_floors=("requested_floor", "nunique"),
        avg_duration=("requested_duration_minutes", "mean"),
        avg_flexibility=("max_time_flexibility_minutes", "mean")
    )
)

print(employee_behavior.describe())
print(employee_behavior.head(20))

# ============================================================
# ROOM BOOKING FREQUENCY BY WORK MODE
# ============================================================

print("\n" + "=" * 60)
print("ROOM BOOKING FREQUENCY BY WORK MODE")
print("=" * 60)

work_mode_analysis = (
    room_booking_requests
    .merge(
        employees[
            ["employee_id", "work_mode", "department"]
        ],
        on="employee_id",
        how="left"
    )
    .groupby("work_mode")
    .agg(
        total_requests=("booking_id", "count"),
        unique_employees=("employee_id", "nunique")
    )
)

work_mode_analysis["requests_per_employee"] = (
    work_mode_analysis["total_requests"]
    / work_mode_analysis["unique_employees"]
)

print(work_mode_analysis)

# ============================================================
# ROOM BOOKING FREQUENCY BY DEPARTMENT
# ============================================================

print("\n" + "=" * 60)
print("ROOM BOOKING FREQUENCY BY DEPARTMENT")
print("=" * 60)

department_analysis = (
    room_booking_requests
    .merge(
        employees[
            ["employee_id", "department"]
        ],
        on="employee_id",
        how="left"
    )
    .groupby("department")
    .agg(
        total_requests=("booking_id", "count"),
        unique_employees=("employee_id", "nunique")
    )
)

department_analysis["requests_per_employee"] = (
    department_analysis["total_requests"]
    / department_analysis["unique_employees"]
)

print(department_analysis.sort_values(
    "requests_per_employee",
    ascending=False
))
