# =============================================================================================================================
#                                           Deployment Inference Pipeline
# ==============================================================================================================================

import json
from pathlib import Path
from bisect import bisect_left
import pandas as pd
from catboost import CatBoostRanker


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Load trained CatBoost ranking model
model = CatBoostRanker()
model.load_model(str(MODEL_DIR / "room_ranking_model.cbm"))


# Load feature schema
with open(MODEL_DIR / "feature_schema.json", "r") as f:
    feature_schema = json.load(f)

FEATURES = feature_schema["features"]


print("Deployment model loaded successfully.")
print("Model features:", len(FEATURES))

# ===================================================================================================================
#         Iska purpose hai: jab new request aaye, app.py employee ki historical behaviour profile lookup kar sake.
# ====================================================================================================================

#----------------------------------------------    Load employee behaviour lookup    ------------------------------------

BEHAVIOR_PATH = MODEL_DIR / "employee_behavior_lookup.csv"

employee_behavior_lookup = pd.read_csv(BEHAVIOR_PATH)

print("Employee behaviour lookup loaded successfully.")
print("Employees in lookup:", employee_behavior_lookup["employee_id"].nunique())


#-------------------------------------------------   Load scarcity lookup tables  ---------------------------------

ROOM_SCARCITY_PATH = MODEL_DIR / "room_scarcity_lookup.csv"
ROOM_TIME_SCARCITY_PATH = MODEL_DIR / "room_time_scarcity_lookup.csv"

room_scarcity_lookup = pd.read_csv(ROOM_SCARCITY_PATH)
room_time_scarcity_lookup = pd.read_csv(ROOM_TIME_SCARCITY_PATH)

print("Scarcity lookup tables loaded successfully.")
print("Rooms:", room_scarcity_lookup["candidate_room_id"].nunique())
print("Room-hour rows:", len(room_time_scarcity_lookup))


# ===========================================================================================================================
#                 Deployment ko new request ke time employees + rooms + existing bookings ka current state chahiye.
#                 Is step mein bas ye datasets app.py mein load karenge.
# ============================================================================================================================

# Load workplace data
DATA_DIR = BASE_DIR / "data" / "raw"

employees = pd.read_csv(DATA_DIR / "employees.csv")
rooms = pd.read_csv(DATA_DIR / "rooms.csv")

room_bookings = pd.read_csv(
    DATA_DIR / "room_bookings.csv",
    parse_dates=[
        "requested_start_datetime",
        "requested_end_datetime",
        "allocated_start_datetime",
        "allocated_end_datetime"
    ]
)

print("Workplace data loaded successfully.")
print("Employees:", len(employees))
print("Rooms:", len(rooms))
print("Existing bookings:", len(room_bookings))

# ===========================================================================================================================
#             New request mein employee_id aayega. Humein us employee ki behaviour row nikalni hai,
#             aur agar employee ke paas historical profile nahi hai, toh cold-start handling karni hogi.
#                           Hum missing employee ko invent ki hui history nahi denge.
# ============================================================================================================================

def get_employee_behavior(employee_id):
    """
    Return the saved behaviour profile for an employee.
    Returns None when the employee has no historical profile.
    """
    profile = employee_behavior_lookup[
        employee_behavior_lookup["employee_id"] == employee_id
    ]

    if profile.empty:
        return None

    return profile.iloc[0].to_dict()


# Quick test
test_employee_id = employee_behavior_lookup.iloc[0]["employee_id"]
test_profile = get_employee_behavior(test_employee_id)

print("Behaviour lookup test successful.")
print("Employee:", test_employee_id)
print("Profile found:", test_profile is not None)


# ============================================================
# PREPARE DETERMINISTIC ALLOCATOR STATE
# ============================================================

# Similar room type hierarchy
similar_room_types = {
    "Conference": ["Meeting", "Training", "Focus"],
    "Meeting": ["Conference", "Training", "Focus"],
    "Training": ["Conference", "Meeting", "Focus"],
    "Focus": ["Meeting", "Conference", "Training"]
}


# Room information lookups
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

# Build room + capacity index
room_capacity_index = {}

for _, room in rooms.iterrows():
    key = (room["room_type"], int(room["floor"]))

    if key not in room_capacity_index:
        room_capacity_index[key] = []

    room_capacity_index[key].append(
        (room["room_id"], int(room["capacity"]))
    )

print("Room capacity index prepared.")
print("Room type-floor combinations:", len(room_capacity_index))


# Floors ordered for hierarchy search
floors = sorted(rooms["floor"].unique())


# Store occupied intervals for each room
occupied_rooms = {
    room["room_id"]: []
    for room in room_records
}

room_last_end = {}


# Use only historically allocated bookings
allocated_history = room_bookings[
    room_bookings["booking_status"] == "Allocated"
].copy()

allocated_history["allocated_start_datetime"] = pd.to_datetime(
    allocated_history["allocated_start_datetime"]
)

allocated_history["allocated_end_datetime"] = pd.to_datetime(
    allocated_history["allocated_end_datetime"]
)


# Populate room occupancy intervals
for _, booking in allocated_history.iterrows():

    room_id = booking["allocated_room_id"]

    occupied_rooms[room_id].append(
        (
            booking["allocated_start_datetime"],
            booking["allocated_end_datetime"]
        )
    )


# Sort intervals by start time
for room_id in occupied_rooms:
    occupied_rooms[room_id].sort()


# Latest allocated booking end for each room
for room_id, intervals in occupied_rooms.items():

    if intervals:
        room_last_end[room_id] = max(
            end for _, end in intervals
        )


print("Deterministic allocator state prepared.")
print("Rooms:", len(room_lookup))
print("Room type-floor combinations:", len(room_capacity_index))
print("Rooms with booking history:", len(room_last_end))

# ===========================================================================================================================
#                      Given room + requested start + duration, room ke existing bookings dekhkar
#                                   batana ki request usi time fit hoti hai ya nahi.
# ============================================================================================================================

# ============================================================
# OPTIMIZED ROOM AVAILABILITY CHECK
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

    # Fast path:
    # If the requested time is after the room's
    # latest booking, the room is immediately available.
    last_end = room_last_end.get(room_id)

    if last_end is None or candidate_start >= last_end:
        return candidate_start

    # Binary search for the first booking whose
    # start time is >= candidate_start.
    index = bisect_left(
        existing_bookings,
        (candidate_start,)
    )

    # Check the booking immediately before the insertion point.
    if index > 0:

        previous_start, previous_end = existing_bookings[index - 1]

        if candidate_start < previous_end:
            candidate_start = previous_end
            index -= 1

    # Jump through conflicting intervals.
    while index < len(existing_bookings):

        existing_start, existing_end = existing_bookings[index]

        # Request fits before this booking.
        if candidate_start + duration <= existing_start:
            break

        # Request starts after this booking.
        if candidate_start >= existing_end:
            index += 1
            continue

        # Conflict → jump directly to booking end.
        candidate_start = existing_end
        index += 1

    return candidate_start


print("Availability engine loaded successfully.")


# ============================================================
# Ye function ek given time par feasible room find karega.
# ============================================================

# ============================================================
# DETERMINISTIC ROOM SEARCH
# ============================================================

def find_room(
    start_datetime,
    duration,
    requested_type,
    requested_floor,
    requested_capacity
):

    floor_order = sorted(
        floors,
        key=lambda floor: (
            abs(floor - requested_floor),
            floor
        )
    )

    # --------------------------------------------------------
    # Check candidate rooms
    # --------------------------------------------------------

    def check_candidates(candidate_rooms, allocation_rule):

        for room_id, room_capacity in candidate_rooms:

            # Capacity is a HARD constraint
            if room_capacity < requested_capacity:
                continue

            next_available = get_next_available_time(
                room_id,
                start_datetime,
                duration,
                occupied_rooms
            )

            # Available exactly at requested candidate time
            if next_available == pd.Timestamp(start_datetime):

                room = room_lookup[room_id]

                return (
                    room,
                    allocation_rule,
                    next_available
                )

        return None


    # ========================================================
    # RULE 1: Exact Type + Exact Floor
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
    # RULE 2: Exact Type + Nearby Floor
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
    # RULE 3: Exact Type + Any Floor
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
    # RULE 4: Similar Type + Exact Floor
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
    # RULE 5: Similar Type + Any Floor
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


    # No room available at this exact candidate time
    return None, "no_room_available", None


print("Deterministic room search loaded successfully.")

# ============================================================
# ROOM SEARCH WITH TIME FLEXIBILITY
# ============================================================

def generate_allowed_start_times(
    requested_start,
    flexibility_minutes
):
    """
    Generate candidate start times in 15-minute increments,
    checking the requested time first, then nearby times.
    """

    allowed_times = []

    requested_start = pd.Timestamp(requested_start)

    for offset in range(
        0,
        int(flexibility_minutes) + 1,
        15
    ):

        if offset == 0:

            allowed_times.append(requested_start)

        else:

            earlier = (
                requested_start
                - pd.Timedelta(minutes=offset)
            )

            later = (
                requested_start
                + pd.Timedelta(minutes=offset)
            )

            allowed_times.append(earlier)
            allowed_times.append(later)

    return sorted(allowed_times)


def find_room_with_flexibility(
    requested_start,
    duration,
    requested_type,
    requested_floor,
    requested_capacity,
    flexibility_minutes
):
    """
    Apply the complete deterministic allocation logic:
    room hierarchy first, then time relaxation.
    """

    allowed_times = generate_allowed_start_times(
        requested_start,
        flexibility_minutes
    )

    for candidate_start in allowed_times:

        result = find_room(
            start_datetime=candidate_start,
            duration=duration,
            requested_type=requested_type,
            requested_floor=requested_floor,
            requested_capacity=requested_capacity
        )

        room, allocation_rule, allocated_start = result

        if room is not None:

            return (
                room,
                allocation_rule,
                allocated_start,
                int(
                    abs(
                        (
                            allocated_start
                            - pd.Timestamp(requested_start)
                        ).total_seconds()
                    ) / 60
                )
            )

    return None, "no_room_available", None, None


print("Time-flexibility engine loaded successfully.")

# ============================================================
# GENERATE FEASIBLE CANDIDATES FOR A NEW REQUEST
# ============================================================

def generate_feasible_candidates_live(
    booking_id,
    employee_id,
    requested_start,
    duration,
    requested_type,
    requested_floor,
    requested_capacity,
    max_time_flexibility
):

    requested_start = pd.Timestamp(requested_start)

    duration = int(duration)
    requested_floor = int(requested_floor)
    requested_capacity = int(requested_capacity)
    max_time_flexibility = int(max_time_flexibility)

    floor_order = sorted(
        floors,
        key=lambda floor: (
            abs(floor - requested_floor),
            floor
        )
    )

    # Try requested time first, then ±15, ±30, ...
    for relaxation in range(
        0,
        max_time_flexibility + 1,
        15
    ):

        if relaxation == 0:

            candidate_times = [
                requested_start
            ]

        else:

            candidate_times = [
                requested_start - pd.Timedelta(
                    minutes=relaxation
                ),
                requested_start + pd.Timedelta(
                    minutes=relaxation
                )
            ]

        for candidate_start in candidate_times:

            candidate_end = (
                candidate_start
                + pd.Timedelta(minutes=duration)
            )

            # ------------------------------------------------
            # Five hierarchy levels
            # ------------------------------------------------

            hierarchy_groups = [

                (
                    "exact_type_exact_floor",
                    [(requested_type, requested_floor)]
                ),

                (
                    "exact_type_nearby_floor",
                    [
                        (requested_type, floor)
                        for floor in floor_order
                        if (
                            floor != requested_floor
                            and abs(floor - requested_floor) == 1
                        )
                    ]
                ),

                (
                    "exact_type_any_floor",
                    [
                        (requested_type, floor)
                        for floor in floor_order
                    ]
                ),

                (
                    "similar_type_exact_floor",
                    [
                        (room_type, requested_floor)
                        for room_type
                        in similar_room_types[requested_type]
                    ]
                ),

                (
                    "similar_type_any_floor",
                    [
                        (room_type, floor)
                        for room_type
                        in similar_room_types[requested_type]
                        for floor in floor_order
                    ]
                )
            ]

            # ------------------------------------------------
            # Find the FIRST hierarchy level with candidates
            # ------------------------------------------------

            for allocation_rule, room_keys in hierarchy_groups:

                level_candidates = []

                for room_type, floor in room_keys:

                    rooms_at_location = room_capacity_index.get(
                        (room_type, floor),
                        []
                    )

                    for room_id, room_capacity in rooms_at_location:

                        # Capacity remains a hard constraint
                        if room_capacity < requested_capacity:
                            continue

                        next_available = get_next_available_time(
                            room_id,
                            candidate_start,
                            duration,
                            occupied_rooms
                        )

                        if next_available == candidate_start:

                            room = room_lookup[room_id]

                            level_candidates.append({
                                "booking_id": booking_id,
                                "employee_id": employee_id,
                                "candidate_room_id": room_id,
                                "candidate_room_type": room["room_type"],
                                "candidate_floor": int(room["floor"]),
                                "candidate_capacity": int(room["capacity"]),
                                "candidate_start_datetime": candidate_start,
                                "candidate_end_datetime": candidate_end,
                                "allocation_rule": allocation_rule,
                                "time_relaxation_minutes": relaxation
                            })

                # Stop immediately at the first successful
                # hierarchy level and time level.
                if level_candidates:

                    return pd.DataFrame(
                        level_candidates
                    )

    # No feasible candidate anywhere within flexibility
    return pd.DataFrame()


print("Corrected live candidate generator loaded successfully.")

# ===============================================================================================================================
#            Hum ek actual booking request lenge, uski details room_booking_requests.csv se lenge,
#                   aur dekhenge ki live generator feasible candidates return karta hai ya nahi.
# ================================================================================================================================

#-----------------------------------------         Load booking request data for testing    ----------------------------------
room_booking_requests = pd.read_csv(
    DATA_DIR / "room_booking_requests.csv",
    parse_dates=[
        "requested_start_datetime",
        "requested_end_datetime"
    ]
)

print("Booking request data loaded successfully.")
print("Requests:", len(room_booking_requests))

# ============================================================
# TEST LIVE CANDIDATE GENERATION
# ============================================================

test_employee_id = employees.iloc[0]["employee_id"]

test_request_time = (
    room_booking_requests["requested_start_datetime"].max()
    + pd.Timedelta(days=1)
).replace(
    hour=10,
    minute=0,
    second=0
)

test_candidates = generate_feasible_candidates_live(
    booking_id="TEST_001",
    employee_id=test_employee_id,
    requested_start=test_request_time,
    duration=60,
    requested_type="Meeting",
    requested_floor=2,
    requested_capacity=8,
    max_time_flexibility=60
)

print("Live candidate generation test completed.")
print("Employee:", test_employee_id)
print("Requested time:", test_request_time)
print("Candidate count:", len(test_candidates))

if not test_candidates.empty:
    print("\nCandidates:")
    print(
        test_candidates[
            [
                "candidate_room_id",
                "candidate_room_type",
                "candidate_floor",
                "candidate_capacity",
                "allocation_rule",
                "time_relaxation_minutes"
            ]
        ].to_string(index=False)
    )
else:
    print("No feasible candidate found.")

    # ============================================================
# ATTACH EMPLOYEE BEHAVIOUR FEATURES
# ============================================================

# ============================================================
# EMPLOYEE BEHAVIOUR FEATURES
# ============================================================

def add_employee_behavior_features(candidate_df):

    # These are created separately from historical
    # employee-candidate relationship logic.
    relationship_columns = {
        "employee_history_count",
        "has_employee_history",
        "employee_candidate_room_count",
        "employee_candidate_room_pct",
        "employee_candidate_type_count",
        "employee_candidate_type_pct",
        "employee_candidate_floor_count",
        "employee_candidate_floor_pct",
        "employee_candidate_hour_count",
        "employee_candidate_hour_pct"
    }

    behavior_columns = [
        column
        for column in employee_behavior_lookup.columns
        if (
            column != "employee_id"
            and column not in relationship_columns
        )
    ]

    behavior_data = employee_behavior_lookup[
        ["employee_id"] + behavior_columns
    ].copy()

    return candidate_df.merge(
        behavior_data,
        on="employee_id",
        how="left",
        validate="many_to_one"
    )


print("Employee behaviour feature builder corrected successfully.")

# ============================================================
# BUILD MODEL FEATURES FOR LIVE CANDIDATES
# ============================================================

def build_live_features(
    candidate_df,
    employee_id,
    requested_start,
    requested_duration,
    requested_type,
    requested_floor,
    requested_capacity
):
    """
    Convert live feasible candidates into the feature structure
    expected by the trained CatBoost ranking model.
    """

    df = candidate_df.copy()

    # Remove all previously created employee-candidate
    # relationship columns, including pandas _x / _y duplicates.

    relationship_bases = {
        "employee_history_count",
        "has_employee_history",
        "employee_candidate_room_count",
        "employee_candidate_room_pct",
        "employee_candidate_type_count",
        "employee_candidate_type_pct",
        "employee_candidate_floor_count",
        "employee_candidate_floor_pct",
        "employee_candidate_hour_count",
        "employee_candidate_hour_pct"
    }

    columns_to_remove = [
        column
        for column in df.columns
        if column in relationship_bases
        or column.endswith("_x")
        or column.endswith("_y")
    ]

    df = df.drop(columns=columns_to_remove)
    

    requested_start = pd.Timestamp(requested_start)

    # --------------------------------------------------------
    # 1. Employee behaviour features
    # --------------------------------------------------------

    df = add_employee_behavior_features(df)

    # --------------------------------------------------------
    # 2. Current request features
    # --------------------------------------------------------

    df["requested_room_type"] = requested_type
    df["requested_floor"] = int(requested_floor)
    df["requested_capacity"] = int(requested_capacity)
    df["requested_duration_minutes"] = int(requested_duration)
    df["requested_start_hour"] = requested_start.hour
    df["requested_start_minute"] = requested_start.minute

    # --------------------------------------------------------
    # 3. Candidate datetime features
    # --------------------------------------------------------

    df["candidate_start_datetime"] = pd.to_datetime(
        df["candidate_start_datetime"]
    )

    df["candidate_end_datetime"] = pd.to_datetime(
        df["candidate_end_datetime"]
    )

    df["candidate_start_hour"] = (
        df["candidate_start_datetime"].dt.hour
    )

    df["candidate_start_minute"] = (
        df["candidate_start_datetime"].dt.minute
    )

    df["candidate_duration_minutes"] = (
        (
            df["candidate_end_datetime"]
            - df["candidate_start_datetime"]
        )
        .dt.total_seconds()
        / 60
    ).astype(int)

    # --------------------------------------------------------
    # 4. Candidate ↔ request relationships
    # --------------------------------------------------------

    df["capacity_difference"] = (
        df["requested_capacity"]
        - df["candidate_capacity"]
    ).abs()

    df["time_difference_minutes"] = (
        (
            df["candidate_start_datetime"]
            - requested_start
        )
        .abs()
        .dt.total_seconds()
        / 60
    ).astype(int)

    # --------------------------------------------------------
    # 5. Historical employee ↔ candidate relationships
    # --------------------------------------------------------

    employee_history = room_bookings[
        (
            room_bookings["booking_status"] == "Allocated"
        )
        &
        (
            room_bookings["employee_id"] == employee_id
        )
    ].copy()

    total_history = len(employee_history)

    df["employee_history_count"] = total_history
    df["has_employee_history"] = int(total_history > 0)

    if total_history > 0:

        room_counts = (
            employee_history["allocated_room_id"]
            .value_counts()
        )

        type_counts = (
            employee_history["allocated_room_type"]
            .value_counts()
        )

        floor_counts = (
            employee_history["allocated_floor"]
            .value_counts()
        )

        hour_counts = (
            pd.to_datetime(
                employee_history["allocated_start_datetime"]
            )
            .dt.hour
            .value_counts()
        )

        df["employee_candidate_room_count"] = (
            df["candidate_room_id"]
            .map(room_counts)
            .fillna(0)
        )

        df["employee_candidate_room_pct"] = (
            df["employee_candidate_room_count"]
            / total_history
        )

        df["employee_candidate_type_count"] = (
            df["candidate_room_type"]
            .map(type_counts)
            .fillna(0)
        )

        df["employee_candidate_type_pct"] = (
            df["employee_candidate_type_count"]
            / total_history
        )

        df["employee_candidate_floor_count"] = (
            df["candidate_floor"]
            .map(floor_counts)
            .fillna(0)
        )

        df["employee_candidate_floor_pct"] = (
            df["employee_candidate_floor_count"]
            / total_history
        )

        df["employee_candidate_hour_count"] = (
            df["candidate_start_hour"]
            .map(hour_counts)
            .fillna(0)
        )

        df["employee_candidate_hour_pct"] = (
            df["employee_candidate_hour_count"]
            / total_history
        )

    else:

        df["employee_candidate_room_count"] = 0
        df["employee_candidate_room_pct"] = 0.0

        df["employee_candidate_type_count"] = 0
        df["employee_candidate_type_pct"] = 0.0

        df["employee_candidate_floor_count"] = 0
        df["employee_candidate_floor_pct"] = 0.0

        df["employee_candidate_hour_count"] = 0
        df["employee_candidate_hour_pct"] = 0.0

    # --------------------------------------------------------
    # 6. Scarcity / resource intelligence
    # --------------------------------------------------------

    room_scarcity = room_scarcity_lookup.set_index(
        "candidate_room_id"
    )

    df["room_historical_demand"] = (
        df["candidate_room_id"]
        .map(room_scarcity["room_historical_demand"])
        .fillna(0)
    )

    df["room_utilization_rate"] = (
        df["candidate_room_id"]
        .map(room_scarcity["room_utilization_rate"])
        .fillna(0)
    )

    # Historical cumulative occupied minutes for each candidate room.
    # This matches the feature used by the trained model.
    room_cumulative_minutes = (
        allocated_history.assign(
            occupied_minutes=(
                pd.to_datetime(allocated_history["allocated_end_datetime"])
                - pd.to_datetime(allocated_history["allocated_start_datetime"])
            ).dt.total_seconds() / 60
        )
        .groupby("allocated_room_id")["occupied_minutes"]
        .sum()
    )

    df["room_cumulative_occupied_minutes"] = (
        df["candidate_room_id"]
        .map(room_cumulative_minutes)
        .fillna(0)
    )

    room_time_key = (
        room_time_scarcity_lookup[
            "candidate_room_id"
        ].astype(str)
        + "_"
        + room_time_scarcity_lookup[
            "candidate_start_hour"
        ].astype(str)
    )

    room_time_demand_lookup = dict(
        zip(
            room_time_key,
            room_time_scarcity_lookup[
                "room_time_demand"
            ]
        )
    )

    df["room_time_demand"] = [
        room_time_demand_lookup.get(
            f"{room_id}_{hour}",
            0
        )
        for room_id, hour in zip(
            df["candidate_room_id"],
            df["candidate_start_hour"]
        )
    ]

    # The saved model was trained with duplicated relationship
    # columns created by a historical merge (_x and _y).
    # They represent the same relationship signals, so reproduce
    # both names at inference time for schema compatibility.
    relationship_features = [
        "employee_history_count",
        "has_employee_history",
        "employee_candidate_room_count",
        "employee_candidate_room_pct",
        "employee_candidate_type_count",
        "employee_candidate_type_pct",
        "employee_candidate_floor_count",
        "employee_candidate_floor_pct",
        "employee_candidate_hour_count",
        "employee_candidate_hour_pct"
    ]

    for feature in relationship_features:
        df[f"{feature}_x"] = df[feature]
        df[f"{feature}_y"] = df[feature]

    # Number of other currently feasible candidates
    df["capacity_feasible_alternative_count"] = (
        len(df) - 1
    )

    # --------------------------------------------------------
    # 7. Ensure model categorical columns are strings
    # --------------------------------------------------------

    df["candidate_room_type"] = (
        df["candidate_room_type"].astype(str)
    )

    df["allocation_rule"] = (
        df["allocation_rule"].astype(str)
    )

    # --------------------------------------------------------
    # 8. Ensure every expected model feature exists
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    # Keep exact training feature order
    model_features = df[FEATURES].copy()

    return model_features


print("Live ML feature builder loaded successfully.")

# ============================================================
# TEST LIVE FEATURES + CATBOOST RANKING
# ============================================================

test_features = build_live_features(
    candidate_df=test_candidates,
    employee_id=test_employee_id,
    requested_start=test_request_time,
    requested_duration=60,
    requested_type="Meeting",
    requested_floor=2,
    requested_capacity=8
)

print("Live feature matrix created.")
print("Feature shape:", test_features.shape)


# Score all feasible candidates
test_scores = model.predict(test_features)

test_results = test_candidates.copy()
test_results["ml_score"] = test_scores

test_results = test_results.sort_values(
    "ml_score",
    ascending=False
).reset_index(drop=True)

print("\nML ranking:")
print(
    test_results[
        [
            "candidate_room_id",
            "candidate_room_type",
            "candidate_floor",
            "candidate_capacity",
            "allocation_rule",
            "time_relaxation_minutes",
            "ml_score"
        ]
    ].to_string(index=False)
)

print("\nRecommended room:", test_results.iloc[0]["candidate_room_id"])

# ============================================================
# FINAL ALLOCATION DECISION
# ============================================================

def allocate_request(candidate_df, employee_id, requested_start,
                     requested_duration, requested_type,
                     requested_floor, requested_capacity):

    # No feasible candidates
    if candidate_df is None or candidate_df.empty:
        return None, "skipped_no_feasible_candidate"

    # Exactly one feasible candidate
    if len(candidate_df) == 1:
        selected = candidate_df.iloc[0].copy()
        selected["ml_score"] = None

        return selected, "direct_allocation"

    # Multiple feasible candidates -> ML ranking
    live_features = build_live_features(
        candidate_df=candidate_df,
        employee_id=employee_id,
        requested_start=requested_start,
        requested_duration=requested_duration,
        requested_type=requested_type,
        requested_floor=requested_floor,
        requested_capacity=requested_capacity
    )

    scores = model.predict(live_features)

    ranked = candidate_df.copy()
    ranked["ml_score"] = scores

    ranked = ranked.sort_values(
        "ml_score",
        ascending=False
    ).reset_index(drop=True)

    selected = ranked.iloc[0].copy()

    return selected, "ml_ranked_allocation"


print("Final allocation decision layer loaded successfully.")


# ========================================================================================================================
#                                                               STREAMLIT UI
# ========================================================================================================================

import streamlit as st

st.set_page_config(
    page_title="Workplace Intelligence",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Workplace Intelligence")
st.subheader("Personalized Meeting Room Recommendation")

st.write(
    "Enter your meeting requirements and the system will "
    "find feasible rooms and rank them using the ML model."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    employee_id = st.text_input(
        "Employee ID",
        value="E0001"
    )

    request_date = st.date_input(
        "Meeting Date"
    )

    start_time = st.time_input(
        "Start Time"
    )

    duration = st.number_input(
        "Duration (minutes)",
        min_value=15,
        max_value=480,
        value=60,
        step=15
    )

with col2:
    room_type = st.selectbox(
        "Room Type",
        ["Meeting", "Conference", "Training", "Focus"]
    )

    floor = st.selectbox(
        "Preferred Floor",
        [1, 2, 3, 4, 5]
    )

    capacity = st.selectbox(
        "Required Capacity",
        [4, 8, 16]
    )

    flexibility = st.selectbox(
        "Time Flexibility (minutes)",
        [0, 15, 30, 45, 60, 90, 120]
    )

st.divider()

button_col = st.columns([1, 2, 1])

with button_col[1]:
    find_room = st.button(
        "Find Best Room",
        type="primary",
        use_container_width=True
    )

if find_room:

    employee_id = employee_id.strip()

    # Validate Employee ID before processing the request
    if employee_id not in set(employees["employee_id"]):
        st.error(
            f"Employee ID '{employee_id}' does not exist. "
            "Please enter a valid Employee ID."
        )
        st.stop()

    requested_start = pd.Timestamp(
        request_date.strftime("%Y-%m-%d") + " " +
        start_time.strftime("%H:%M:%S")
    )

    # Generate feasible candidates
    candidates = generate_feasible_candidates_live(
        booking_id="LIVE_REQUEST",
        employee_id=employee_id,
        requested_start=requested_start,
        duration=duration,
        requested_type=room_type,
        requested_floor=floor,
        requested_capacity=capacity,
        max_time_flexibility=flexibility
    )

    # ============================================================
    # NO FEASIBLE ROOM
    # ============================================================

    if candidates is None or candidates.empty:

        st.error(
            "No feasible room is available for this request."
        )

    # ============================================================
    # FEASIBLE ROOMS FOUND
    # ============================================================

    else:

        selected, decision = allocate_request(
            candidate_df=candidates,
            employee_id=employee_id,
            requested_start=requested_start,
            requested_duration=duration,
            requested_type=room_type,
            requested_floor=floor,
            requested_capacity=capacity
        )

        if selected is not None:

            st.success("Room successfully recommended.")

            # ====================================================
            # RECOMMENDED ROOM
            # ====================================================

            st.subheader("🏆 Recommended Room")

            result_col1, result_col2, result_col3, result_col4 = st.columns(4)

            with result_col1:
                st.metric(
                    "Room",
                    selected["candidate_room_id"]
                )

            with result_col2:
                st.metric(
                    "Room Type",
                    selected["candidate_room_type"]
                )

            with result_col3:
                st.metric(
                    "Floor",
                    int(selected["candidate_floor"])
                )

            with result_col4:
                st.metric(
                    "Capacity",
                    int(selected["candidate_capacity"])
                )

            st.write(
                "**Allocated Start:**",
                pd.Timestamp(
                    selected["candidate_start_datetime"]
                ).strftime("%d %b %Y, %H:%M")
            )

            st.write(
                "**Decision:**",
                decision.replace("_", " ").title()
            )

            st.write(
                "**Allocation Rule:**",
                selected["allocation_rule"].replace("_", " ").title()
            )

            if selected["ml_score"] is not None:
                st.write(
                    "**ML Ranking Score:**",
                    round(float(selected["ml_score"]), 4)
                )

            # ====================================================
            # CANDIDATE RANKING
            # ====================================================

            if len(candidates) > 1:

                st.divider()
                st.subheader("📊 Candidate Ranking")

                ranking_table = candidates.copy()

                ranking_features = build_live_features(
                    candidate_df=candidates,
                    employee_id=employee_id,
                    requested_start=requested_start,
                    requested_duration=duration,
                    requested_type=room_type,
                    requested_floor=floor,
                    requested_capacity=capacity
                )

                ranking_scores = model.predict(
                    ranking_features[FEATURES]
                )

                ranking_table["ml_score"] = ranking_scores

                ranking_table = ranking_table.sort_values(
                    "ml_score",
                    ascending=False
                ).reset_index(drop=True)

                ranking_table["Rank"] = range(
                    1,
                    len(ranking_table) + 1
                )

                ranking_table["candidate_start_datetime"] = (
                    pd.to_datetime(
                        ranking_table["candidate_start_datetime"]
                    ).dt.strftime("%d %b %H:%M")
                )

                ranking_table = ranking_table[
                    [
                        "Rank",
                        "candidate_room_id",
                        "candidate_room_type",
                        "candidate_floor",
                        "candidate_capacity",
                        "candidate_start_datetime",
                        "allocation_rule",
                        "time_relaxation_minutes",
                        "ml_score"
                    ]
                ]

                ranking_table = ranking_table.rename(
                    columns={
                        "candidate_room_id": "Room",
                        "candidate_room_type": "Type",
                        "candidate_floor": "Floor",
                        "candidate_capacity": "Capacity",
                        "candidate_start_datetime": "Start",
                        "allocation_rule": "Allocation Rule",
                        "time_relaxation_minutes": "Time Flexibility Used",
                        "ml_score": "ML Score"
                    }
                )

                ranking_table["ML Score"] = (
                    ranking_table["ML Score"].round(4)
                )

                st.dataframe(
                    ranking_table,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.error(
                "No feasible room could be selected."
            )