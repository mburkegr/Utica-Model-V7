import os
from functools import lru_cache
from datetime import date

import numpy as np
import pandas as pd

try:
    import pyxirr
except Exception:
    pyxirr = None


# -----------------------------
# Library + utility helpers
# -----------------------------
def clean_tc_name(name):
    return str(name).strip().lower().replace(" ", "_")


@lru_cache(maxsize=8)
def _load_type_curve_library_cached(file_path, file_mtime):
    tc_monthly = pd.read_excel(file_path, sheet_name="tc_monthly")
    tc_metadata = pd.read_excel(file_path, sheet_name="tc_metadata")

    tc_monthly["tc_name"] = tc_monthly["tc_name"].map(clean_tc_name)
    tc_metadata["tc_name"] = tc_metadata["tc_name"].map(clean_tc_name)

    library = {}
    for tc_name, monthly_df in tc_monthly.groupby("tc_name"):
        meta_row = tc_metadata.loc[tc_metadata["tc_name"] == tc_name]
        if meta_row.empty:
            continue

        library[tc_name] = {
            "base_lateral": float(meta_row["base_lateral"].iloc[0]),
            "monthly": monthly_df[["month", "oil", "gas"]].copy().sort_values("month"),
        }

    return library


def load_type_curve_library(file_path="type_curve_library.xlsx"):
    file_mtime = os.path.getmtime(file_path)
    return _load_type_curve_library_cached(file_path, file_mtime)

@lru_cache(maxsize=8)
def _load_price_file_cached(file_path, file_mtime):
    """Read and validate the monthly oil and gas pricing file."""

    pricing_df = pd.read_excel(
        file_path,
        sheet_name="Data",
    )

    if pricing_df.empty:
        raise ValueError(
            "The pricing file does not contain any pricing rows."
        )
    
    required_columns = {
        "month",
        "oil_price",
        "gas_price",
    }

    missing_columns = required_columns - set(pricing_df.columns)

    if missing_columns:
        raise ValueError(
            "Pricing file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    pricing_df = pricing_df[
        [
            "month",
            "oil_price",
            "gas_price",
        ]
    ].copy()

    # This works whether Excel stores the month as an actual date
    # or as text such as 01/01/2026.
    pricing_df["month"] = pd.to_datetime(
        pricing_df["month"],
        errors="coerce",
    )

    pricing_df["oil_price"] = pd.to_numeric(
        pricing_df["oil_price"],
        errors="coerce",
    )

    pricing_df["gas_price"] = pd.to_numeric(
        pricing_df["gas_price"],
        errors="coerce",
    )

    if pricing_df["month"].isna().any():
        raise ValueError(
            "The pricing file contains one or more invalid months."
        )

    if pricing_df[["oil_price", "gas_price"]].isna().any().any():
        raise ValueError(
            "The pricing file contains one or more invalid oil or gas prices."
        )

    # Normalize every date to the first day of its month.
    pricing_df["month"] = (
        pricing_df["month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    pricing_df = (
        pricing_df
        .sort_values("month")
        .reset_index(drop=True)
    )

    duplicate_months = pricing_df.loc[
        pricing_df["month"].duplicated(keep=False),
        "month",
    ]

    if not duplicate_months.empty:
        duplicate_text = ", ".join(
            duplicate_months.dt.strftime("%b %Y").unique()
        )

        raise ValueError(
            f"The pricing file contains duplicate months: {duplicate_text}"
        )

    # Verify that there are no missing months inside the deck.
    expected_months = pd.date_range(
        start=pricing_df["month"].min(),
        end=pricing_df["month"].max(),
        freq="MS",
    )

    actual_months = pd.DatetimeIndex(pricing_df["month"])

    missing_months = expected_months.difference(actual_months)

    if not missing_months.empty:
        missing_text = ", ".join(
            missing_months.strftime("%b %Y")
        )

        raise ValueError(
            f"The pricing file is missing monthly pricing for: {missing_text}"
        )

    return pricing_df


def load_price_file(file_path="price_file_library.xlsx"):
    """Load the pricing file and refresh the cache when GitHub updates it."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Pricing file was not found: {file_path}"
        )

    file_mtime = os.path.getmtime(file_path)

    return _load_price_file_cached(
        file_path,
        file_mtime,
    ).copy()

def default_effective_date():
    today = date.today()
    if today.month == 12:
        return pd.Timestamp(today.year + 1, 1, 1)
    return pd.Timestamp(today.year, today.month + 1, 1)

def build_index_price_series(
    dates,
    global_assumptions,
):
    """
    Build the monthly WTI and Henry Hub pricing series.

    Flat mode:
        Uses the app-entered oil and gas prices for every month.

    File mode:
        Uses the pricing file before each commodity's switch date.
        Uses the app-entered flat price beginning with the switch month.
    """

    output = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates))
    })

    output["date"] = (
        output["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    pricing_mode = str(
        global_assumptions.get("pricing_mode", "flat")
    ).lower()

    terminal_oil_price = float(
        global_assumptions["oil_price"]
    )

    terminal_gas_price = float(
        global_assumptions["gas_price"]
    )

    # Preserve your current flat-pricing behavior.
    if pricing_mode == "flat":
        output["index_oil_price"] = terminal_oil_price
        output["index_gas_price"] = terminal_gas_price

        return output[
            [
                "date",
                "index_oil_price",
                "index_gas_price",
            ]
        ]

    if pricing_mode != "file":
        raise ValueError(
            f"Unsupported pricing mode: {pricing_mode}"
        )

    pricing_file_path = global_assumptions.get(
        "pricing_file_path",
        "price_file_library.xlsx",
    )

    pricing_deck = load_price_file(
        pricing_file_path
    ).rename(
        columns={
            "month": "date",
            "oil_price": "deck_oil_price",
            "gas_price": "deck_gas_price",
        }
    )

    output = output.merge(
        pricing_deck,
        on="date",
        how="left",
    )

    oil_flat_start_date = (
        pd.Timestamp(
            global_assumptions["oil_flat_start_date"]
        )
        .to_period("M")
        .to_timestamp()
    )

    gas_flat_start_date = (
        pd.Timestamp(
            global_assumptions["gas_flat_start_date"]
        )
        .to_period("M")
        .to_timestamp()
    )

    # These base values stay unchanged during sensitivities.
    base_oil_price = float(
        global_assumptions.get(
            "base_oil_price",
            terminal_oil_price,
        )
    )

    base_gas_price = float(
        global_assumptions.get(
            "base_gas_price",
            terminal_gas_price,
        )
    )

    # This allows your existing sensitivity functions to shift
    # the entire pricing curve.
    oil_curve_shift = (
        terminal_oil_price - base_oil_price
    )

    gas_curve_shift = (
        terminal_gas_price - base_gas_price
    )

    output["index_oil_price"] = np.where(
        output["date"] >= oil_flat_start_date,
        terminal_oil_price,
        output["deck_oil_price"] + oil_curve_shift,
    )

    output["index_gas_price"] = np.where(
        output["date"] >= gas_flat_start_date,
        terminal_gas_price,
        output["deck_gas_price"] + gas_curve_shift,
    )

    # Only require file pricing before the applicable flat date.
    missing_oil_mask = (
        output["date"].lt(oil_flat_start_date)
        & output["index_oil_price"].isna()
    )

    missing_gas_mask = (
        output["date"].lt(gas_flat_start_date)
        & output["index_gas_price"].isna()
    )

    if missing_oil_mask.any():
        missing_dates = output.loc[
            missing_oil_mask,
            "date",
        ].dt.strftime("%b %Y")

        raise ValueError(
            "The pricing file does not contain oil pricing for "
            + ", ".join(missing_dates.unique()[:5])
            + ". Either add those months to the file or select "
            "an earlier oil flat-pricing date."
        )

    if missing_gas_mask.any():
        missing_dates = output.loc[
            missing_gas_mask,
            "date",
        ].dt.strftime("%b %Y")

        raise ValueError(
            "The pricing file does not contain gas pricing for "
            + ", ".join(missing_dates.unique()[:5])
            + ". Either add those months to the file or select "
            "an earlier gas flat-pricing date."
        )

    return output[
        [
            "date",
            "index_oil_price",
            "index_gas_price",
        ]
    ]

# -----------------------------
# NGL factors
# -----------------------------
def build_slot_ngl_factors(
    slot,
    global_assumptions,
    content_percentages,
    recover_ethane_percentages,
    reject_ethane_percentages,
    ngl_prices,
    ngl_shrink_factors,
):
    ngl_yield = float(slot["ngl_yield"])

    if int(global_assumptions["ethane_rec"]) == 1:
        recoveries = recover_ethane_percentages
        recovery_case = "recover"
    else:
        recoveries = reject_ethane_percentages
        recovery_case = "reject"

    rows = []

    for component in content_percentages:
        content_pct = float(content_percentages[component])
        implied_ngl_content = content_pct * ngl_yield
        recovery_pct = float(recoveries[component])
        sales_volume_factor = implied_ngl_content * recovery_pct
        shrink_factor = float(ngl_shrink_factors[component])
        shrink_contribution = sales_volume_factor * shrink_factor
        component_price = float(ngl_prices[component])

        rows.append(
            {
                "component": component,
                "content_pct": content_pct,
                "implied_ngl_content": implied_ngl_content,
                "recovery_pct": recovery_pct,
                "sales_volume_factor": sales_volume_factor,
                "shrink_factor": shrink_factor,
                "shrink_contribution": shrink_contribution,
                "component_price": component_price,
            }
        )

    ngl_detail_df = pd.DataFrame(rows)

    shrink = float(ngl_detail_df["shrink_contribution"].sum())

    aggregate_ngl_price = float(
        (
            ngl_detail_df["recovery_pct"]
            * ngl_detail_df["content_pct"]
            * ngl_detail_df["component_price"]
        ).sum()
    )

    ngl_pct_of_wti = aggregate_ngl_price * 42.0 / 70.0

    return {
        "recovery_case": recovery_case,
        "recoveries": recoveries,
        "detail_df": ngl_detail_df,
        "shrink": shrink,
        "aggregate_ngl_price": aggregate_ngl_price,
        "ngl_pct_of_wti": ngl_pct_of_wti,
    }


# -----------------------------
# Slot metrics
# -----------------------------
def calc_slot_metrics(slot, deal_settings, total_net_acres):
    slot = slot.copy()

    if bool(deal_settings["use_bid_override"]):
        bid_price_final = max(1.0, float(deal_settings["bid_override"]))
    else:
        bid_price_final = max(1.0, float(slot["bid_per_acre"]))

    if bool(slot["use_calc_unit_acres"]):
        unit_acres_final = (
            float(slot["gross_wells"]) * float(slot["lateral_length"]) / 50.0
        )
    else:
        unit_acres_final = float(slot["unit_acres"])

    if unit_acres_final == 0:
        working_interest = 0.0
    else:
        working_interest = (
            float(slot["net_acres"]) / unit_acres_final
        ) * float(slot["pct_unitized"])

    net_wells = working_interest * float(slot["gross_wells"])

    use_acquisition_override = bool(
        deal_settings.get("use_acquisition_override", False)
    )
    acquisition_cost_override = float(
        deal_settings.get("acquisition_cost_override", 0.0)
    )

    if use_acquisition_override:
        if total_net_acres == 0:
            acquisition_cost = 0.0
        else:
            acquisition_cost = (
                float(slot["net_acres"]) / total_net_acres
            ) * acquisition_cost_override
    else:
        acquisition_cost = float(slot["net_acres"]) * bid_price_final

    slot["bid_price_final"] = bid_price_final
    slot["unit_acres_final"] = unit_acres_final
    slot["working_interest"] = working_interest
    slot["net_wells_calc"] = net_wells
    slot["acquisition_cost"] = acquisition_cost

    return slot


# -----------------------------
# Single well economics
# -----------------------------
def run_single_slot_economics(slot, type_curve_library, global_assumptions, slot_ngl):
    tc_name = clean_tc_name(slot["tc_name"])
    lateral_length = float(slot["lateral_length"])
    spud_date = pd.to_datetime(slot["drilling_spud_month"])
    flowback_delay = int(slot["flowback_delay"])
    ngl_yield = float(slot["ngl_yield"])
    nri = float(slot["net_revenue_interest"])
    tc_risk = float(slot["tc_risk"])


    use_sev_tax_pct = bool(
        global_assumptions.get("use_sev_tax_pct", False)
    )
    
    oil_sev_tax = float(global_assumptions["oil_sev_tax"])
    gas_sev_tax = float(global_assumptions["gas_sev_tax"])
    ad_val_tax = float(global_assumptions["ad_val_tax"])

    oil_diff = float(slot["oil_diff"])
    gas_diff = float(slot["gas_diff"])

    oil_opex_bbl = float(slot["oil_opex_bbl"])
    gas_opex_mcf = float(slot["gas_opex_mcf"])
    ngl_opex = float(slot["ngl_opex"])
    fixed_loe = float(slot["fixed_loe"])
    dc_costs = float(slot["dc_costs"])

    tc_info = type_curve_library[tc_name]
    base_lateral = float(tc_info["base_lateral"])
    
    tc_df = (
        tc_info["monthly"]
        .copy()
        .sort_values("month")
        .reset_index(drop=True)
    )
    
    tc_df["period"] = pd.to_numeric(
        tc_df["month"],
        errors="raise",
    ).astype(int)
    
    spud_month = (
        pd.Timestamp(spud_date)
        .to_period("M")
        .to_timestamp()
    )
    
    production_start_month = (
        spud_month
        + pd.DateOffset(months=flowback_delay)
    )
    
    tc_df["date"] = [
        production_start_month
        + pd.DateOffset(months=int(period) - 1)
        for period in tc_df["period"]
    ]
    
    index_pricing = build_index_price_series(
        dates=tc_df["date"],
        global_assumptions=global_assumptions,
    )
    
    tc_df["index_oil_price"] = (
        index_pricing["index_oil_price"].to_numpy()
    )
    
    tc_df["index_gas_price"] = (
        index_pricing["index_gas_price"].to_numpy()
    )
    
    ll_scale = lateral_length / base_lateral

    tc_df["base_oil_scaled"] = tc_df["oil"] * tc_risk * ll_scale
    tc_df["base_gas_scaled"] = tc_df["gas"] * tc_risk * ll_scale

    tc_df["gross_oil_production"] = tc_df["base_oil_scaled"]
    tc_df["gross_gas_production"] = tc_df["base_gas_scaled"] * (
        1.0 - float(slot_ngl["shrink"])
    )
    tc_df["gross_ngl_production"] = tc_df["base_gas_scaled"] * ngl_yield / 42.0

    tc_df["monthly_production_boe"] = (
        tc_df["gross_oil_production"]
        + tc_df["gross_ngl_production"]
        + (tc_df["gross_gas_production"] / 6.0)
    )

    tc_df["oil_royalty_volumes"] = tc_df["gross_oil_production"] * (1.0 - nri)
    tc_df["gas_royalty_volumes"] = tc_df["gross_gas_production"] * (1.0 - nri)
    tc_df["ngl_royalty_volumes"] = tc_df["gross_ngl_production"] * (1.0 - nri)

    tc_df["equity_oil_production"] = (
        tc_df["gross_oil_production"] - tc_df["oil_royalty_volumes"]
    )
    tc_df["equity_gas_production"] = (
        tc_df["gross_gas_production"] - tc_df["gas_royalty_volumes"]
    )
    tc_df["equity_ngl_production"] = (
        tc_df["gross_ngl_production"] - tc_df["ngl_royalty_volumes"]
    )

    tc_df["local_oil_price"] = (
        tc_df["index_oil_price"]
        + oil_diff
    )
    
    tc_df["local_gas_price"] = (
        tc_df["index_gas_price"]
        + gas_diff
    )
    
    tc_df["local_ngl_price"] = (
        tc_df["index_oil_price"]
        * float(slot_ngl["ngl_pct_of_wti"])
    )

    tc_df["oil_revenue"] = tc_df["local_oil_price"] * tc_df["equity_oil_production"]
    tc_df["gas_revenue"] = tc_df["local_gas_price"] * tc_df["equity_gas_production"]
    tc_df["ngl_revenue"] = tc_df["local_ngl_price"] * tc_df["equity_ngl_production"]
    tc_df["total_revenue"] = (
        tc_df["oil_revenue"] + tc_df["gas_revenue"] + tc_df["ngl_revenue"]
    )

    variable_loe = -(
        tc_df["gross_oil_production"] * oil_opex_bbl
        + tc_df["gross_gas_production"] * gas_opex_mcf
        + tc_df["gross_ngl_production"] * ngl_opex
    )
    fixed_loe_monthly = -fixed_loe

    tc_df["variable_loe"] = variable_loe
    tc_df["fixed_loe_monthly"] = fixed_loe_monthly
    tc_df["total_loe"] = tc_df["variable_loe"] + tc_df["fixed_loe_monthly"]

    tc_df["ad_valorem_tax"] = -(
        ad_val_tax * tc_df["total_revenue"]
    )
    
    if use_sev_tax_pct:
        # Net product revenue equals net production multiplied
        # by the realized product price.
        tc_df["oil_severance_tax"] = -(
            oil_sev_tax * tc_df["oil_revenue"]
        )
    
        tc_df["gas_severance_tax"] = -(
            gas_sev_tax * tc_df["gas_revenue"]
        )
    
    else:
        # Preserve the existing fixed-rate methodology.
        tc_df["oil_severance_tax"] = -(
            oil_sev_tax * tc_df["equity_oil_production"]
        )
    
        tc_df["gas_severance_tax"] = -(
            gas_sev_tax
            * (
                tc_df["equity_gas_production"]
                / (1.0 - float(slot_ngl["shrink"]))
            )
        )

    tc_df["tax"] = (
        tc_df["ad_valorem_tax"]
        + tc_df["oil_severance_tax"]
        + tc_df["gas_severance_tax"]
    )

    tc_df["net_revenue"] = tc_df["total_revenue"]
    tc_df["opex"] = tc_df["variable_loe"]

    period_0_pricing = build_index_price_series(
        dates=[spud_month],
        global_assumptions=global_assumptions,
    ).iloc[0]
    
    period_0_oil_index = float(
        period_0_pricing["index_oil_price"]
    )
    
    period_0_gas_index = float(
        period_0_pricing["index_gas_price"]
    )
    
    period_0 = pd.DataFrame(
        {
            "period": [0],
            "date": [spud_month],
            "index_oil_price": [period_0_oil_index],
            "index_gas_price": [period_0_gas_index],
            "base_oil_scaled": [0.0],
            "base_gas_scaled": [0.0],
            "gross_oil_production": [0.0],
            "gross_gas_production": [0.0],
            "gross_ngl_production": [0.0],
            "monthly_production_boe": [0.0],
            "oil_royalty_volumes": [0.0],
            "gas_royalty_volumes": [0.0],
            "ngl_royalty_volumes": [0.0],
            "equity_oil_production": [0.0],
            "equity_gas_production": [0.0],
            "equity_ngl_production": [0.0],
            "local_oil_price": [
                period_0_oil_index + oil_diff
            ],
            "local_gas_price": [
                period_0_gas_index + gas_diff
            ],
            "local_ngl_price": [
                period_0_oil_index
                * float(slot_ngl["ngl_pct_of_wti"])
            ],
            "oil_revenue": [0.0],
            "gas_revenue": [0.0],
            "ngl_revenue": [0.0],
            "total_revenue": [0.0],
            "net_revenue": [0.0],
            "opex": [0.0],
            "variable_loe": [0.0],
            "fixed_loe_monthly": [0.0],
            "total_loe": [0.0],
            "tax": [0.0],
        }
    )
    
    df = pd.concat(
        [
            period_0,
            tc_df[
                [
                    "date",
                    "index_oil_price",
                    "index_gas_price",
                    "period",
                    "base_oil_scaled",
                    "base_gas_scaled",
                    "gross_oil_production",
                    "gross_gas_production",
                    "gross_ngl_production",
                    "monthly_production_boe",
                    "oil_royalty_volumes",
                    "gas_royalty_volumes",
                    "ngl_royalty_volumes",
                    "equity_oil_production",
                    "equity_gas_production",
                    "equity_ngl_production",
                    "local_oil_price",
                    "local_gas_price",
                    "local_ngl_price",
                    "oil_revenue",
                    "gas_revenue",
                    "ngl_revenue",
                    "total_revenue",
                    "net_revenue",
                    "opex",
                    "variable_loe",
                    "fixed_loe_monthly",
                    "total_loe",
                    "tax",
                ]
            ],
        ],
        ignore_index=True,
    )
    df = df.sort_values("period").reset_index(drop=True)

    df["capex"] = 0.0
    df.loc[df["period"] == 0, "capex"] = -(dc_costs * lateral_length)

    # Determine the economic limit using the unmodified operating cash flow.
    # Once a producing well has a negative operating month, it is permanently
    # shut in beginning in that month. This prevents late-life fixed LOE from
    # creating a second cash-flow sign change and unstable/multiple XIRRs.
    df["pre_shut_in_operating_cf"] = (
        df["net_revenue"] + df["total_loe"] + df["tax"]
    )

    df["economic_limit_reached"] = (
        df["period"].gt(0)
        & df["pre_shut_in_operating_cf"].lt(0.0)
    )
    df["well_shut_in"] = df["economic_limit_reached"].cummax()

    shut_in_cols = [
        "base_oil_scaled",
        "base_gas_scaled",
        "gross_oil_production",
        "gross_gas_production",
        "gross_ngl_production",
        "monthly_production_boe",
        "oil_royalty_volumes",
        "gas_royalty_volumes",
        "ngl_royalty_volumes",
        "equity_oil_production",
        "equity_gas_production",
        "equity_ngl_production",
        "oil_revenue",
        "gas_revenue",
        "ngl_revenue",
        "total_revenue",
        "net_revenue",
        "opex",
        "variable_loe",
        "fixed_loe_monthly",
        "total_loe",
        "tax",
    ]
    existing_shut_in_cols = [c for c in shut_in_cols if c in df.columns]
    df.loc[df["well_shut_in"], existing_shut_in_cols] = 0.0

    # Recalculate the final operating cash flow after applying the permanent
    # shut-in. Keep operating_cf_shut_in for backward-compatible outputs.
    df["operating_cf"] = df["net_revenue"] + df["total_loe"] + df["tax"]
    df["operating_cf_shut_in"] = df["operating_cf"]
    df["cash_flow"] = df["operating_cf"] + df["capex"]

    return df


# -----------------------------
# Slot financials
# -----------------------------

def build_slot_financials(
    slot, deal_settings, type_curve_library, global_assumptions, total_net_acres
):
    slot = calc_slot_metrics(slot, deal_settings, total_net_acres)

    slot_ngl = build_slot_ngl_factors(
        slot=slot,
        global_assumptions=global_assumptions,
        content_percentages=global_assumptions["content_percentages"],
        recover_ethane_percentages=global_assumptions["recover_ethane_percentages"],
        reject_ethane_percentages=global_assumptions["reject_ethane_percentages"],
        ngl_prices=global_assumptions["ngl_prices"],
        ngl_shrink_factors=global_assumptions["ngl_shrink_factors"],
    )

    one_well_df = run_single_slot_economics(
        slot=slot,
        type_curve_library=type_curve_library,
        global_assumptions=global_assumptions,
        slot_ngl=slot_ngl,
    ).copy()

    required_cols = [
        "index_oil_price",
        "index_gas_price",
        "gross_oil_production",
        "gross_gas_production",
        "gross_ngl_production",
        "monthly_production_boe",
        "equity_oil_production",
        "equity_gas_production",
        "equity_ngl_production",
        "local_oil_price",
        "local_gas_price",
        "local_ngl_price",
        "total_loe",
        "tax",
        "operating_cf",
        "capex",
    ]
    missing_cols = [c for c in required_cols if c not in one_well_df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in one_well_df: {missing_cols}")

    df = one_well_df.copy()

    gross_wells = float(slot["gross_wells"])
    base_working_interest = float(slot["working_interest"])
    base_net_wells = float(slot["net_wells_calc"])
    lease_nri = float(slot["net_revenue_interest"])

    # USEDC funds D&C at the combined USEDC + Granite WI, then Granite's
    # carried percentage is removed from USEDC's production ownership.
    carry_enabled = bool(slot.get("carry_enabled", False))
    carry_wi_reversion_pct = (
        float(slot.get("carry_wi_reversion_pct", 0.0))
        if carry_enabled
        else 0.0
    )
    carry_wi_reversion_pct = float(
        np.clip(carry_wi_reversion_pct, 0.0, 1.0)
    )
    post_carry_ownership_factor = 1.0 - carry_wi_reversion_pct

    dale_promote = bool(slot.get("dale_promote", False))
    dale_unit_id = str(slot.get("dale_unit_id", f"UNIT-{int(slot['slot_id'])}")).strip()
    dale_payout_group = str(
        slot.get("dale_payout_group", dale_unit_id)
    ).strip()
    dale_first_well_carry = bool(
        slot.get("dale_first_well_carry", False)
    )

    # Model-basis assumption:
    # base_working_interest is the ORIGINAL WI represented by the slot before
    # either Dale's initial 1/16 interest or the USEDC-to-Granite carry split.
    # Dale's initial interest is therefore taken directly from original WI.
    dale_initial_interest_pct = float(
        np.clip(
            deal_settings.get("dale_initial_interest_pct", 0.0625),
            0.0,
            1.0,
        )
    )
    if dale_promote:
        dale_initial_working_interest = (
            base_working_interest * dale_initial_interest_pct
        )
    else:
        dale_initial_working_interest = 0.0

    gr_parties_working_interest = (
        base_working_interest - dale_initial_working_interest
    )
    gr_parties_working_interest = max(
        0.0,
        gr_parties_working_interest,
    )
    gr_parties_net_wells = (
        gr_parties_working_interest * gross_wells
    )

    # A checked row represents the first well in that physical unit. Even when
    # the slot contains several gross wells, only one well receives Dale's
    # initial-interest D&C carry.
    dale_first_well_carry_wells = (
        min(1.0, max(0.0, gross_wells))
        if dale_promote and dale_first_well_carry
        else 0.0
    )
    dale_carry_dnc_net_wells = (
        dale_initial_working_interest
        * dale_first_well_carry_wells
    )
    funded_dnc_net_wells = (
        gr_parties_net_wells + dale_carry_dnc_net_wells
    )

    df["slot_id"] = slot["slot_id"]
    df["tc_name"] = slot["tc_name"]
    df["dale_promote"] = dale_promote
    df["dale_unit_id"] = dale_unit_id
    df["dale_payout_group"] = dale_payout_group
    df["dale_first_well_carry"] = dale_first_well_carry
    df["dale_initial_interest_pct"] = dale_initial_interest_pct
    df["pre_dale_working_interest"] = base_working_interest
    df["dale_initial_working_interest"] = dale_initial_working_interest
    df["post_initial_dale_working_interest"] = gr_parties_working_interest
    df["gr_parties_working_interest"] = gr_parties_working_interest
    df["gr_parties_net_wells"] = gr_parties_net_wells
    df["dale_first_well_carry_wells"] = dale_first_well_carry_wells
    df["dale_carry_dnc_net_wells"] = dale_carry_dnc_net_wells
    df["funded_dnc_net_wells"] = funded_dnc_net_wells

    # First-production Granite carry:
    # Period 0 is the spud / D&C month.
    # Period 1 is the first modeled production month.
    df["carry_reversion_active"] = (
        carry_enabled & df["period"].gt(0)
    )
    df["ownership_factor"] = np.where(
        df["carry_reversion_active"],
        post_carry_ownership_factor,
        1.0,
    )
    df["effective_working_interest"] = (
        gr_parties_working_interest * df["ownership_factor"]
    )
    df["effective_net_wells"] = (
        gr_parties_net_wells * df["ownership_factor"]
    )

    # Helpful carry audit fields. The Granite carry is applied only after
    # Dale's initial interest has first been removed from original WI.
    df["pre_carry_working_interest"] = gr_parties_working_interest
    df["post_carry_working_interest"] = (
        gr_parties_working_interest * post_carry_ownership_factor
    )
    df["pre_carry_effective_nri"] = (
        gr_parties_working_interest * lease_nri
    )
    df["post_carry_effective_nri"] = (
        gr_parties_working_interest
        * post_carry_ownership_factor
        * lease_nri
    )
    df["pre_carry_net_wells"] = gr_parties_net_wells
    df["post_carry_net_wells"] = (
        gr_parties_net_wells * post_carry_ownership_factor
    )

    df["slot_gross_oil_production"] = df["gross_oil_production"] * gross_wells
    df["slot_gross_gas_production"] = df["gross_gas_production"] * gross_wells
    df["slot_gross_ngl_production"] = df["gross_ngl_production"] * gross_wells
    df["slot_gross_boe"] = df["monthly_production_boe"] * gross_wells

    # USEDC return cash flow uses USEDC's retained ownership after the Granite
    # carry. The Dale payout OCF is calculated separately below using the full
    # combined USEDC + Granite interest.
    df["slot_net_oil_production"] = (
        df["equity_oil_production"] * df["effective_net_wells"]
    )
    df["slot_net_gas_production"] = (
        df["equity_gas_production"] * df["effective_net_wells"]
    )
    df["slot_net_ngl_production"] = (
        df["equity_ngl_production"] * df["effective_net_wells"]
    )
    df["slot_net_boe"] = (
        df["equity_oil_production"]
        + df["equity_ngl_production"]
        + (df["equity_gas_production"] / 6.0)
    ) * df["effective_net_wells"]

    df["slot_oil_revenue"] = (
        df["equity_oil_production"]
        * df["local_oil_price"]
        * df["effective_net_wells"]
    )
    df["slot_gas_revenue"] = (
        df["equity_gas_production"]
        * df["local_gas_price"]
        * df["effective_net_wells"]
    )
    df["slot_ngl_revenue"] = (
        df["equity_ngl_production"]
        * df["local_ngl_price"]
        * df["effective_net_wells"]
    )
    df["slot_total_revenue"] = (
        df["slot_oil_revenue"]
        + df["slot_gas_revenue"]
        + df["slot_ngl_revenue"]
    )
    df["slot_loe"] = df["total_loe"] * df["effective_net_wells"]
    df["slot_tax"] = df["tax"] * df["effective_net_wells"]
    df["slot_operating_profit"] = (
        df["slot_total_revenue"] + df["slot_loe"] + df["slot_tax"]
    )

    # Dale payout is intentionally based on positive OCF after LOE and taxes,
    # but before the USEDC-to-Granite carry split. It uses the combined USEDC +
    # Granite / GR Parties interest after Dale's initial 1/16 has been removed.
    df["slot_promote_ocf"] = (
        df["operating_cf"] * gr_parties_net_wells
    )

    # Base D&C covers the combined USEDC + Granite interest for every well
    # after Dale's initial ownership is removed. Dale's initial 1/16 D&C is
    # added only for the flagged first well in each physical unit.
    df["slot_base_capex"] = (
        df["capex"] * gr_parties_net_wells
    )
    df["slot_dale_carry_capex"] = (
        df["capex"] * dale_carry_dnc_net_wells
    )
    df["slot_capex"] = (
        df["slot_base_capex"] + df["slot_dale_carry_capex"]
    )

    df["slot_pud_cash_flow"] = (
        df["slot_operating_profit"] + df["slot_capex"]
    )
    df["slot_asset_purchase"] = 0.0
    df["slot_promote"] = 0.0
    df["slot_total_cash_flow"] = (
        df["slot_pud_cash_flow"]
        + df["slot_asset_purchase"]
        + df["slot_promote"]
    )

    # Preserve original ownership fields for backward-compatible outputs.
    df["working_interest"] = base_working_interest
    df["net_wells"] = base_net_wells
    df["gross_wells"] = gross_wells
    df["acquisition_cost"] = float(slot["acquisition_cost"])
    df["bid_price_final"] = float(slot["bid_price_final"])
    df["ngl_recovery_case"] = slot_ngl["recovery_case"]
    df["slot_shrink"] = float(slot_ngl["shrink"])
    df["slot_ngl_pct_of_wti"] = float(slot_ngl["ngl_pct_of_wti"])

    return df

def align_to_financial_calendar(slot_df, effective_date, months=360):
    effective_date = (
        pd.to_datetime(effective_date).to_period("M").to_timestamp()
    )

    slot_df = slot_df.copy()
    slot_df["date"] = (
        pd.to_datetime(slot_df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    slot_start = slot_df["date"].min()
    calendar_start = (
        min(effective_date, slot_start)
        if pd.notnull(slot_start)
        else effective_date
    )
    calendar_end = effective_date + pd.DateOffset(months=months - 1)

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                start=calendar_start,
                end=calendar_end,
                freq="MS",
            )
        }
    )

    df = calendar.merge(slot_df, on="date", how="left")

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns if col != "slot_id"
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if col == "tc_name":
            df[col] = df[col].fillna(
                slot_df["tc_name"].iloc[0] if "tc_name" in slot_df.columns else ""
            )
        elif col == "ngl_recovery_case":
            df[col] = df[col].fillna(
                slot_df["ngl_recovery_case"].iloc[0]
                if "ngl_recovery_case" in slot_df.columns
                else ""
            )

    if "slot_id" in slot_df.columns:
        df["slot_id"] = slot_df["slot_id"].iloc[0]

    return df


# -----------------------------
# Deal build / rollup
# -----------------------------
def build_all_slot_financials(
    slot_inputs, deal_settings, type_curve_library, global_assumptions
):
    slot_results = []
    total_net_acres = pd.to_numeric(
        slot_inputs["net_acres"], errors="coerce"
    ).fillna(0).sum()

    effective_date = pd.to_datetime(deal_settings["effective_date"])

    for _, slot_row in slot_inputs.iterrows():
        slot_df = build_slot_financials(
            slot=slot_row,
            deal_settings=deal_settings,
            type_curve_library=type_curve_library,
            global_assumptions=global_assumptions,
            total_net_acres=total_net_acres,
        )

        slot_df = align_to_financial_calendar(
            slot_df,
            deal_settings["effective_date"],
            months=360,
        )

        # Repopulate the monthly index prices across the entire
        # aligned 360-month calendar.
        aligned_index_pricing = build_index_price_series(
            dates=slot_df["date"],
            global_assumptions=global_assumptions,
        )
        
        slot_df = (
            slot_df
            .drop(
                columns=[
                    "index_oil_price",
                    "index_gas_price",
                ],
                errors="ignore",
            )
            .merge(
                aligned_index_pricing,
                on="date",
                how="left",
            )
        )

        # These are slot-level attributes and must remain populated on every
        # aligned calendar row, including the acquisition month before spud.
        slot_df["dale_promote"] = bool(slot_row.get("dale_promote", False))
        slot_df["dale_unit_id"] = str(
            slot_row.get("dale_unit_id", f"UNIT-{int(slot_row['slot_id'])}")
        ).strip()
        slot_df["dale_payout_group"] = str(
            slot_row.get("dale_payout_group", slot_df["dale_unit_id"].iloc[0])
        ).strip()
        slot_df["dale_first_well_carry"] = bool(
            slot_row.get("dale_first_well_carry", False)
        )
        slot_df["carry_enabled"] = bool(slot_row.get("carry_enabled", False))

        slot_calc = calc_slot_metrics(slot_row, deal_settings, total_net_acres)

        slot_df["slot_asset_purchase"] = 0.0
        mask = (
            (slot_df["date"].dt.year == effective_date.year)
            & (slot_df["date"].dt.month == effective_date.month)
        )
        slot_df.loc[mask, "slot_asset_purchase"] = -float(
            slot_calc["acquisition_cost"]
        )

        if "slot_promote" not in slot_df.columns:
            slot_df["slot_promote"] = 0.0

        slot_df["slot_total_cash_flow"] = (
            slot_df["slot_pud_cash_flow"]
            + slot_df["slot_asset_purchase"]
            + slot_df["slot_promote"]
        )

        slot_results.append(slot_df)

    return pd.concat(slot_results, ignore_index=True)



def roll_up_deal(all_slots_df):
    sum_cols = [
        "slot_gross_oil_production",
        "slot_gross_gas_production",
        "slot_gross_ngl_production",
        "slot_gross_boe",
        "slot_net_oil_production",
        "slot_net_gas_production",
        "slot_net_ngl_production",
        "slot_net_boe",
        "slot_oil_revenue",
        "slot_gas_revenue",
        "slot_ngl_revenue",
        "slot_total_revenue",
        "slot_loe",
        "slot_tax",
        "slot_operating_profit",
        "slot_promote_ocf",
        "slot_base_capex",
        "slot_dale_carry_capex",
        "slot_capex",
        "slot_pud_cash_flow",
        "slot_asset_purchase",
        "slot_total_cash_flow",
    ]

    existing_sum_cols = [c for c in sum_cols if c in all_slots_df.columns]
    deal_df = (
        all_slots_df.groupby("date", as_index=False)[existing_sum_cols]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Each payout group has its own hurdle schedule. Build a non-duplicated
    # portfolio summary for the deal-level audit; activation still occurs at
    # the individual payout-group level in apply_promote_to_slots().
    schedule_cols = [
        "promote_monthly_investment",
        "promote_monthly_distributions",
        "promote_cumulative_investment",
        "promote_cumulative_distributions",
        "promote_running_multiple",
        "promote_hurdle_reached",
        "promote_active",
        "promote_hurdle_date",
        "promote_effective_date",
    ]
    required = {"dale_payout_group", "date", *schedule_cols}

    if required.issubset(all_slots_df.columns):
        group_schedule = (
            all_slots_df.loc[
                all_slots_df["dale_promote"].fillna(False),
                ["dale_payout_group", "date", *schedule_cols],
            ]
            .drop_duplicates(["dale_payout_group", "date"])
            .copy()
        )

        if not group_schedule.empty:
            numeric_cols = [
                "promote_monthly_investment",
                "promote_monthly_distributions",
                "promote_cumulative_investment",
                "promote_cumulative_distributions",
            ]
            summary = (
                group_schedule.groupby("date", as_index=False)[numeric_cols]
                .sum()
                .sort_values("date")
            )

            bool_summary = (
                group_schedule.groupby("date", as_index=False)[
                    ["promote_hurdle_reached", "promote_active"]
                ]
                .any()
            )
            summary = summary.merge(bool_summary, on="date", how="left")

            summary["promote_running_multiple"] = np.where(
                summary["promote_cumulative_investment"] > 0.0,
                summary["promote_cumulative_distributions"]
                / summary["promote_cumulative_investment"],
                0.0,
            )

            first_dates = (
                group_schedule.groupby("dale_payout_group", as_index=False)[
                    ["promote_hurdle_date", "promote_effective_date"]
                ]
                .first()
            )
            earliest_hurdle = pd.to_datetime(
                first_dates["promote_hurdle_date"], errors="coerce"
            ).min()
            earliest_effective = pd.to_datetime(
                first_dates["promote_effective_date"], errors="coerce"
            ).min()
            summary["promote_hurdle_date"] = earliest_hurdle
            summary["promote_effective_date"] = earliest_effective

            active_counts = (
                group_schedule.groupby("date")["promote_active"]
                .sum()
                .rename("promote_active_group_count")
                .reset_index()
            )
            total_groups = int(group_schedule["dale_payout_group"].nunique())
            summary = summary.merge(active_counts, on="date", how="left")
            summary["promote_total_group_count"] = total_groups

            deal_df = deal_df.merge(summary, on="date", how="left")

    return deal_df


def build_promote_schedule(promoted_rows, deal_settings):
    """Build a separate Dale payout schedule for each unit or well tranche.

    The denominator is acquisition cost plus all D&C funded by USEDC,
    including the flagged first-well D&C carry for Dale's initial interest.
    The numerator is positive OCF after LOE and taxes for the combined USEDC +
    Granite / GR Parties interest, before the USEDC-to-Granite carry split.
    The additional WI back-in becomes effective in the following modeled month.
    """
    group_col = "dale_payout_group"
    monthly = (
        promoted_rows.groupby([group_col, "date"], as_index=False)[
            ["slot_promote_ocf", "slot_capex", "slot_asset_purchase"]
        ]
        .sum()
        .sort_values([group_col, "date"])
        .reset_index(drop=True)
    )

    monthly["promote_monthly_investment"] = (
        -monthly["slot_asset_purchase"].clip(upper=0.0)
        -monthly["slot_capex"].clip(upper=0.0)
    )
    monthly["promote_monthly_distributions"] = monthly[
        "slot_promote_ocf"
    ].clip(lower=0.0)

    monthly["promote_cumulative_investment"] = monthly.groupby(group_col)[
        "promote_monthly_investment"
    ].cumsum()
    monthly["promote_cumulative_distributions"] = monthly.groupby(group_col)[
        "promote_monthly_distributions"
    ].cumsum()

    invested = monthly["promote_cumulative_investment"]
    monthly["promote_running_multiple"] = np.where(
        invested > 0.0,
        monthly["promote_cumulative_distributions"] / invested,
        0.0,
    )

    hurdle = float(deal_settings.get("promote_multiple", 0.0))
    monthly["promote_hurdle_reached"] = (
        (invested > 0.0)
        & (monthly["promote_running_multiple"] >= hurdle)
    )

    vested = monthly.groupby(group_col)[
        "promote_hurdle_reached"
    ].cummax()
    monthly["promote_active"] = (
        vested.groupby(monthly[group_col])
        .shift(1, fill_value=False)
        .astype(bool)
    )

    hurdle_dates = (
        monthly.loc[monthly["promote_hurdle_reached"]]
        .groupby(group_col)["date"]
        .min()
    )
    effective_dates = (
        monthly.loc[monthly["promote_active"]]
        .groupby(group_col)["date"]
        .min()
    )
    monthly["promote_hurdle_date"] = monthly[group_col].map(hurdle_dates)
    monthly["promote_effective_date"] = monthly[group_col].map(effective_dates)

    return monthly[
        [
            group_col,
            "date",
            "promote_monthly_investment",
            "promote_monthly_distributions",
            "promote_cumulative_investment",
            "promote_cumulative_distributions",
            "promote_running_multiple",
            "promote_hurdle_reached",
            "promote_active",
            "promote_hurdle_date",
            "promote_effective_date",
        ]
    ]


def apply_promote_to_slots(all_slots_df, deal_settings):
    """Apply each payout group's permanent Dale WI back-in."""
    df = all_slots_df.copy().sort_values(["date", "slot_id"]).reset_index(drop=True)

    if "dale_promote" not in df.columns:
        df["dale_promote"] = False
    if "dale_payout_group" not in df.columns:
        df["dale_payout_group"] = (
            "SLOT-" + pd.to_numeric(df["slot_id"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )

    df["dale_promote"] = df["dale_promote"].fillna(False).astype(bool)
    df["dale_payout_group"] = df["dale_payout_group"].fillna("").astype(str)
    df["slot_promote"] = 0.0

    # Baseline ownership is USEDC's WI after the Granite carry but before the
    # additional Dale back-in. Therefore the extra 6.25% is shared fairly by
    # USEDC and Granite: USEDC loses 6.25% of its own then-current WI.
    df["pre_promote_working_interest"] = df["effective_working_interest"]
    df["pre_promote_net_wells"] = df["effective_net_wells"]
    df["promote_ownership_factor"] = 1.0
    df["promote_wi_transferred"] = 0.0
    df["post_promote_working_interest"] = df["pre_promote_working_interest"]
    df["post_promote_net_wells"] = df["pre_promote_net_wells"]

    schedule_defaults = {
        "promote_monthly_investment": 0.0,
        "promote_monthly_distributions": 0.0,
        "promote_cumulative_investment": 0.0,
        "promote_cumulative_distributions": 0.0,
        "promote_running_multiple": 0.0,
        "promote_hurdle_reached": False,
        "promote_active": False,
        "promote_hurdle_date": pd.NaT,
        "promote_effective_date": pd.NaT,
    }
    for col, default in schedule_defaults.items():
        df[col] = default

    promoted_mask = df["dale_promote"]
    promote_enabled = bool(deal_settings.get("promote_enabled", False))

    if promote_enabled and promoted_mask.any():
        schedule = build_promote_schedule(
            df.loc[promoted_mask].copy(),
            deal_settings,
        )

        schedule_cols = [
            c for c in schedule.columns
            if c not in {"dale_payout_group", "date"}
        ]
        df = df.drop(columns=schedule_cols, errors="ignore").merge(
            schedule,
            on=["dale_payout_group", "date"],
            how="left",
        )

        numeric_schedule_cols = [
            "promote_monthly_investment",
            "promote_monthly_distributions",
            "promote_cumulative_investment",
            "promote_cumulative_distributions",
            "promote_running_multiple",
        ]
        df[numeric_schedule_cols] = df[numeric_schedule_cols].fillna(0.0)
        df["promote_hurdle_reached"] = (
            df["promote_hurdle_reached"].fillna(False).astype(bool)
        )
        df["promote_active"] = df["promote_active"].fillna(False).astype(bool)

        wi_reversion_pct = float(
            np.clip(
                deal_settings.get("promote_wi_reversion_pct", 0.0),
                0.0,
                1.0,
            )
        )
        post_promote_factor = 1.0 - wi_reversion_pct
        active_promoted_mask = df["dale_promote"] & df["promote_active"]

        df.loc[active_promoted_mask, "promote_ownership_factor"] = (
            post_promote_factor
        )
        df.loc[active_promoted_mask, "promote_wi_transferred"] = (
            df.loc[active_promoted_mask, "pre_promote_working_interest"]
            * wi_reversion_pct
        )
        df["post_promote_working_interest"] = (
            df["pre_promote_working_interest"]
            * df["promote_ownership_factor"]
        )
        df["post_promote_net_wells"] = (
            df["pre_promote_net_wells"]
            * df["promote_ownership_factor"]
        )

        # Scale USEDC's ownership economics after the back-in. Dale's separate
        # first-well carry capex is an obligation and is not itself an ownership
        # interest, so only base D&C is scaled if future capex occurs after vesting.
        scale_cols = [
            "slot_net_oil_production",
            "slot_net_gas_production",
            "slot_net_ngl_production",
            "slot_net_boe",
            "slot_oil_revenue",
            "slot_gas_revenue",
            "slot_ngl_revenue",
            "slot_total_revenue",
            "slot_loe",
            "slot_tax",
            "slot_operating_profit",
            "slot_base_capex",
        ]
        existing_scale_cols = [c for c in scale_cols if c in df.columns]
        df.loc[active_promoted_mask, existing_scale_cols] = (
            df.loc[active_promoted_mask, existing_scale_cols]
            .multiply(post_promote_factor)
        )

        if {"slot_base_capex", "slot_dale_carry_capex"}.issubset(df.columns):
            df["slot_capex"] = (
                df["slot_base_capex"] + df["slot_dale_carry_capex"]
            )

        df.loc[active_promoted_mask, "effective_working_interest"] = df.loc[
            active_promoted_mask, "post_promote_working_interest"
        ]
        df.loc[active_promoted_mask, "effective_net_wells"] = df.loc[
            active_promoted_mask, "post_promote_net_wells"
        ]
        df.loc[active_promoted_mask, "ownership_factor"] = (
            df.loc[active_promoted_mask, "ownership_factor"]
            * post_promote_factor
        )

    df["slot_pud_cash_flow"] = df["slot_operating_profit"] + df["slot_capex"]
    df["slot_total_cash_flow"] = (
        df["slot_pud_cash_flow"] + df["slot_asset_purchase"]
    )

    return df.sort_values(["slot_id", "date"]).reset_index(drop=True)

def calc_financial_irr(df):
    if pyxirr is None:
        return None
    try:
        return float(pyxirr.xirr(df["date"], df["slot_total_cash_flow"]))
    except Exception:
        return None


def calc_financial_moic(df):
    invested = -df.loc[df["slot_total_cash_flow"] < 0, "slot_total_cash_flow"].sum()
    returned = df.loc[df["slot_total_cash_flow"] > 0, "slot_total_cash_flow"].sum()

    if invested == 0:
        return None

    return float(returned / invested)


# -----------------------------
# Input prep
# -----------------------------

def prepare_deal_settings(deal_inputs):
    effective_date = pd.to_datetime(
        deal_inputs.get("effective_date", default_effective_date())
    )

    promote_enabled = bool(deal_inputs.get("promote_enabled", False))

    if "promote_wi_reversion_pct" in deal_inputs:
        promote_wi_reversion_pct = (
            float(deal_inputs.get("promote_wi_reversion_pct", 0.0)) / 100.0
        )
    else:
        promote_wi_reversion_pct = float(deal_inputs.get("promote_rate", 0.0))
        if promote_wi_reversion_pct > 1.0:
            promote_wi_reversion_pct /= 100.0

    promote_wi_reversion_pct = float(
        np.clip(promote_wi_reversion_pct, 0.0, 1.0)
    )

    # Whole-percent app input: 6.25 means Dale initially owns 1/16 of the
    # acquired lease. The slot WI is assumed to be the remaining GR-side WI.
    dale_initial_interest_pct = float(
        np.clip(
            float(deal_inputs.get("dale_initial_interest_pct", 6.25)) / 100.0,
            0.0,
            0.999999,
        )
    )

    return {
        "effective_date": effective_date,
        "use_bid_override": bool(deal_inputs.get("use_bid_override", False)),
        "bid_override": max(1.0, float(deal_inputs.get("bid_override", 1.0))),
        "use_acquisition_override": bool(
            deal_inputs.get("use_acquisition_override", False)
        ),
        "acquisition_cost_override": float(
            deal_inputs.get("acquisition_cost_override", 0.0)
        ),
        "promote_enabled": promote_enabled,
        "dale_initial_interest_pct": dale_initial_interest_pct,
        "promote_wi_reversion_pct": (
            promote_wi_reversion_pct if promote_enabled else 0.0
        ),
        "promote_multiple": (
            float(deal_inputs.get("promote_multiple", 0.0))
            if promote_enabled
            else 0.0
        ),
    }

def prepare_global_assumptions(deal_inputs):
    use_sev_tax_pct = bool(
        deal_inputs.get("use_sev_tax_pct", False)
    )

    pricing_mode = str(
        deal_inputs.get("pricing_mode", "flat")
    ).lower()

    if pricing_mode not in {"flat", "file"}:
        raise ValueError(
            f"Unsupported pricing mode: {pricing_mode}"
        )

    return {
        "pricing_mode": pricing_mode,

        "pricing_file_path": str(
            deal_inputs.get(
                "pricing_file_path",
                "price_file_library.xlsx",
            )
        ),

        # In flat mode, these are the flat prices.
        # In file mode, these are the terminal flat prices.
        "oil_price": float(deal_inputs["oil_price"]),
        "gas_price": float(deal_inputs["gas_price"]),

        # These retain the original base-case values during sensitivities.
        "base_oil_price": float(
            deal_inputs.get(
                "base_oil_price",
                deal_inputs["oil_price"],
            )
        ),

        "base_gas_price": float(
            deal_inputs.get(
                "base_gas_price",
                deal_inputs["gas_price"],
            )
        ),

        "oil_flat_start_date": pd.to_datetime(
            deal_inputs.get(
                "oil_flat_start_date",
                "1900-01-01",
            )
        ),

        "gas_flat_start_date": pd.to_datetime(
            deal_inputs.get(
                "gas_flat_start_date",
                "1900-01-01",
            )
        ),

        "use_sev_tax_pct": use_sev_tax_pct,

        # In percentage mode, the app accepts whole percentages:
        # 5 = 5%, so convert it to 0.05 for the calculation.
        "oil_sev_tax": (
            float(deal_inputs["oil_sev_tax"]) / 100.0
            if use_sev_tax_pct
            else float(deal_inputs["oil_sev_tax"])
        ),

        "gas_sev_tax": (
            float(deal_inputs["gas_sev_tax"]) / 100.0
            if use_sev_tax_pct
            else float(deal_inputs["gas_sev_tax"])
        ),

        "ad_val_tax": float(deal_inputs["ad_val_tax"]),
        "ethane_rec": 1 if bool(deal_inputs["ethane_rec"]) else 0,
        "content_percentages": {
            "ethane": float(deal_inputs["content_ethane"]),
            "propane": float(deal_inputs["content_propane"]),
            "isobutane": float(deal_inputs["content_isobutane"]),
            "butane": float(deal_inputs["content_butane"]),
            "pentanes": float(deal_inputs["content_pentanes"]),
        },
        "recover_ethane_percentages": {
            "ethane": float(deal_inputs["rec_ethane"]),
            "propane": float(deal_inputs["rec_propane"]),
            "isobutane": float(deal_inputs["rec_isobutane"]),
            "butane": float(deal_inputs["rec_butane"]),
            "pentanes": float(deal_inputs["rec_pentanes"]),
        },
        "reject_ethane_percentages": {
            "ethane": float(deal_inputs["rej_ethane"]),
            "propane": float(deal_inputs["rej_propane"]),
            "isobutane": float(deal_inputs["rej_isobutane"]),
            "butane": float(deal_inputs["rej_butane"]),
            "pentanes": float(deal_inputs["rej_pentanes"]),
        },
        "ngl_prices": {
            "ethane": float(deal_inputs["price_ethane"]),
            "propane": float(deal_inputs["price_propane"]),
            "isobutane": float(deal_inputs["price_isobutane"]),
            "butane": float(deal_inputs["price_butane"]),
            "pentanes": float(deal_inputs["price_pentanes"]),
        },
        "ngl_shrink_factors": {
            "ethane": float(deal_inputs["shrink_ethane"]),
            "propane": float(deal_inputs["shrink_propane"]),
            "isobutane": float(deal_inputs["shrink_isobutane"]),
            "butane": float(deal_inputs["shrink_butane"]),
            "pentanes": float(deal_inputs["shrink_pentanes"]),
        },
    }



def prepare_slot_inputs(slot_df, deal_inputs):
    df = slot_df.copy()

    required_defaults = {
        "slot_id": 0,
        "dale_promote": False,
        "dale_unit_id": "",
        "dale_payout_group": "",
        "dale_first_well_carry": False,
        "carry_enabled": False,
        "carry_wi_reversion_pct": 0.0,
        "use_calc_unit_acres": False,
        "flowback_delay": 4,
        "tc_risk": 1.0,
        "oil_diff": 0.0,
        "gas_diff": 0.0,
        "ngl_diff": 0.0,
        "oil_opex_bbl": 0.0,
        "gas_opex_mcf": 0.0,
        "ngl_opex": 0.0,
        "fixed_loe": 0.0,
        "ngl_yield": 0.0,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    if "drilling_spud_month" in df.columns:
        df["drilling_spud_month"] = pd.to_datetime(df["drilling_spud_month"])
    else:
        df["drilling_spud_month"] = pd.Timestamp(default_effective_date())

    if bool(deal_inputs.get("use_dc_override", False)):
        df["dc_costs"] = float(deal_inputs.get("dc_override", 0.0))
    else:
        df["dc_costs"] = df["dc_costs"].astype(float)

    if bool(deal_inputs.get("use_bid_override", False)):
        df["bid_per_acre"] = max(
            1.0,
            float(deal_inputs.get("bid_override", 1.0)),
        )
    else:
        df["bid_per_acre"] = (
            pd.to_numeric(df["bid_per_acre"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)
        )

    # Deal-level carry override. This is intentionally applied before the
    # whole-percent carry input is converted to a decimal below, so entering
    # 20 here is exactly the same as entering 20 in every slot's carry field.
    if bool(deal_inputs.get("use_carry_override", False)):
        df["carry_enabled"] = True
        df["carry_wi_reversion_pct"] = float(
            deal_inputs.get("carry_override_pct", 0.0)
        )

    numeric_cols = [
        "slot_id",
        "carry_wi_reversion_pct",
        "lateral_length",
        "gross_wells",
        "net_acres",
        "unit_acres",
        "pct_unitized",
        "flowback_delay",
        "net_revenue_interest",
        "dc_costs",
        "tc_risk",
        "bid_per_acre",
        "oil_diff",
        "gas_diff",
        "ngl_diff",
        "oil_opex_bbl",
        "gas_opex_mcf",
        "ngl_opex",
        "fixed_loe",
        "ngl_yield",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["bid_per_acre"] = df["bid_per_acre"].clip(lower=1.0)
    df["use_calc_unit_acres"] = df["use_calc_unit_acres"].astype(bool)
    df["tc_name"] = df["tc_name"].astype(str)
    df["dale_promote"] = df["dale_promote"].astype(bool)
    df["dale_first_well_carry"] = df["dale_first_well_carry"].astype(bool)
    df["carry_enabled"] = df["carry_enabled"].astype(bool)

    df["carry_wi_reversion_pct"] = (
        pd.to_numeric(df["carry_wi_reversion_pct"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
        / 100.0
    )

    default_group = (
        "UNIT-"
        + pd.to_numeric(df["slot_id"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    df["dale_unit_id"] = df["dale_unit_id"].fillna("").astype(str).str.strip()
    df.loc[df["dale_unit_id"].eq(""), "dale_unit_id"] = default_group

    df["dale_payout_group"] = (
        df["dale_payout_group"].fillna("").astype(str).str.strip()
    )
    df.loc[df["dale_payout_group"].eq(""), "dale_payout_group"] = df.loc[
        df["dale_payout_group"].eq(""), "dale_unit_id"
    ]

    return df


def build_slot_audit_view(all_slots_df):
    df = all_slots_df.copy().sort_values(["slot_id", "date"]).reset_index(drop=True)

    df["month_label"] = df["date"].dt.strftime("%Y-%m")
    df["cum_slot_total_cf"] = df.groupby("slot_id")["slot_total_cash_flow"].cumsum()

    audit_cols = [
        "slot_id",
        "tc_name",
        "date",
        "month_label",
        "period",
        "index_oil_price",
        "index_gas_price",
        "gross_wells",
        "net_wells",
        "working_interest",
        "pre_dale_working_interest",
        "dale_promote",
        "dale_unit_id",
        "dale_payout_group",
        "dale_first_well_carry",
        "dale_initial_interest_pct",
        "dale_initial_working_interest",
        "post_initial_dale_working_interest",
        "gr_parties_working_interest",
        "gr_parties_net_wells",
        "dale_first_well_carry_wells",
        "dale_carry_dnc_net_wells",
        "funded_dnc_net_wells",
        "carry_reversion_active",
        "pre_carry_working_interest",
        "post_carry_working_interest",
        "economic_limit_reached",
        "well_shut_in",
        "pre_shut_in_operating_cf",
        "slot_promote_ocf",
        "pre_promote_working_interest",
        "promote_wi_transferred",
        "post_promote_working_interest",
        "effective_working_interest",
        "pre_promote_net_wells",
        "post_promote_net_wells",
        "effective_net_wells",
        "promote_monthly_investment",
        "promote_monthly_distributions",
        "promote_cumulative_investment",
        "promote_cumulative_distributions",
        "promote_running_multiple",
        "promote_hurdle_reached",
        "promote_active",
        "promote_hurdle_date",
        "promote_effective_date",
        "bid_price_final",
        "acquisition_cost",
        "slot_shrink",
        "slot_ngl_pct_of_wti",
        "slot_gross_oil_production",
        "slot_gross_gas_production",
        "slot_gross_ngl_production",
        "slot_gross_boe",
        "slot_net_oil_production",
        "slot_net_gas_production",
        "slot_net_ngl_production",
        "slot_net_boe",
        "slot_oil_revenue",
        "slot_gas_revenue",
        "slot_ngl_revenue",
        "slot_total_revenue",
        "slot_loe",
        "slot_tax",
        "slot_operating_profit",
        "slot_base_capex",
        "slot_dale_carry_capex",
        "slot_capex",
        "slot_asset_purchase",
        "slot_total_cash_flow",
        "cum_slot_total_cf",
    ]

    existing_cols = [c for c in audit_cols if c in df.columns]
    return df[existing_cols]


def build_deal_audit_view(deal_df):
    df = deal_df.copy().sort_values("date").reset_index(drop=True)

    df["month_num"] = np.arange(1, len(df) + 1)
    df["month_label"] = df["date"].dt.strftime("%Y-%m")
    df["cum_total_cf"] = df["slot_total_cash_flow"].cumsum()

    audit_cols = [
        "date",
        "month_label",
        "month_num",
        "index_oil_price",
        "index_gas_price",
        "promote_monthly_investment",
        "promote_monthly_distributions",
        "promote_cumulative_investment",
        "promote_cumulative_distributions",
        "promote_running_multiple",
        "promote_hurdle_reached",
        "promote_active",
        "promote_active_group_count",
        "promote_total_group_count",
        "promote_hurdle_date",
        "promote_effective_date",
        "slot_gross_oil_production",
        "slot_gross_gas_production",
        "slot_gross_ngl_production",
        "slot_gross_boe",
        "slot_net_oil_production",
        "slot_net_gas_production",
        "slot_net_ngl_production",
        "slot_net_boe",
        "slot_oil_revenue",
        "slot_gas_revenue",
        "slot_ngl_revenue",
        "slot_total_revenue",
        "slot_loe",
        "slot_tax",
        "slot_operating_profit",
        "slot_promote_ocf",
        "slot_base_capex",
        "slot_dale_carry_capex",
        "slot_capex",
        "slot_asset_purchase",
        "slot_total_cash_flow",
        "cum_total_cf",
    ]

    existing_cols = [c for c in audit_cols if c in df.columns]
    return df[existing_cols]

def run_deal_metrics(slot_df, deal_inputs, type_curve_file="type_curve_library.xlsx"):
    """Run the financial engine and return only IRR and MOIC.

    This is the lightweight path used by sensitivities and scenario charts.
    It intentionally skips the audit-view construction and deal-level pricing
    merge that are only needed for the detailed base-case outputs.
    """
    type_curve_library = load_type_curve_library(type_curve_file)
    slot_inputs = prepare_slot_inputs(slot_df, deal_inputs)
    deal_settings = prepare_deal_settings(deal_inputs)
    global_assumptions = prepare_global_assumptions(deal_inputs)

    all_slots_df = build_all_slot_financials(
        slot_inputs=slot_inputs,
        deal_settings=deal_settings,
        type_curve_library=type_curve_library,
        global_assumptions=global_assumptions,
    )
    all_slots_df = apply_promote_to_slots(all_slots_df, deal_settings)
    deal_df = roll_up_deal(all_slots_df)

    return calc_financial_irr(deal_df), calc_financial_moic(deal_df)


def run_deal_model(slot_df, deal_inputs, type_curve_file="type_curve_library.xlsx"):
    type_curve_library = load_type_curve_library(type_curve_file)
    slot_inputs = prepare_slot_inputs(slot_df, deal_inputs)
    deal_settings = prepare_deal_settings(deal_inputs)
    global_assumptions = prepare_global_assumptions(deal_inputs)

    all_slots_df = build_all_slot_financials(
        slot_inputs=slot_inputs,
        deal_settings=deal_settings,
        type_curve_library=type_curve_library,
        global_assumptions=global_assumptions,
    )

    all_slots_df = apply_promote_to_slots(all_slots_df, deal_settings)
    
    deal_df = roll_up_deal(all_slots_df)
    
    deal_index_pricing = build_index_price_series(
        dates=deal_df["date"],
        global_assumptions=global_assumptions,
    )
    
    deal_df = deal_df.merge(
        deal_index_pricing,
        on="date",
        how="left",
    )
    
    slot_audit_df = build_slot_audit_view(all_slots_df)
    deal_audit_df = build_deal_audit_view(deal_df)

    irr = calc_financial_irr(deal_df)
    moic = calc_financial_moic(deal_df)

    return all_slots_df, deal_df, slot_audit_df, deal_audit_df, irr, moic
