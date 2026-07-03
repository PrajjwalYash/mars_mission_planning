import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from complete_year_analysis import (
    export_complete_year_irradiance,
    plot_available_energy_wo_dust,
    plot_daily_maxima,
)
from complete_year_irr import complete_year_irradiance
from dust_deposition_analysis import (
    avail_energy_2028,
    avail_energy_2031,
    dd_fac_2028,
    dd_fac_2031,
    get_dd_ref,
)
from load_support import load_support_duration
from mars_environment import (
    all_wavelength_dust_od,
    am_mars_spectrum,
    atm_gas,
    get_optical_depth,
    optical_params,
    surface_albedo,
    water_ice_cloud,
)


# -----------------------------------------------------------------------------
# Application constants
# -----------------------------------------------------------------------------

LS = np.linspace(0, 355, 72)
BATTERY_EFFICIENCY = 0.83
PANEL_EFFICIENCY = 0.28
MISSION_SOLS = np.arange(1, 181, 1)
SUPPORT_SOLS = np.arange(1, 180, 1)

SITES = [
    {"site_name": "elysium", "lat": 3, "full_name": "Elysium Planitia 3 N, 136 E "},
    {"site_name": "oxia", "lat": 18.75, "full_name": "Oxia Planum 18 N, 325 E "},
    {"site_name": "mawrth_vallis", "lat": 22.3, "full_name": "Mawrth Vallis 22 N, 343 E "},
    {"site_name": "vernal", "lat": 6, "full_name": "Vernal Crater 6 N, 355 E "},
    {"site_name": "valles", "lat": -13.9, "full_name": "Valles Marineres 14 S, 300 E "},
    {"site_name": "aram", "lat": 2.6, "full_name": "Aram Chaos 2 N, 339 E "},
    {"site_name": "meridiani", "lat": -1.95, "full_name": "Meridiani Planum 2 S, 354 E "},
    {"site_name": "eberswalde", "lat": -24, "full_name": "Eberswalde Crater 24 S, 327 E "},
]


# -----------------------------------------------------------------------------
# Presentation helpers
# -----------------------------------------------------------------------------

def apply_page_style() -> None:
    """Apply restrained dashboard styling without affecting chart data."""
    st.markdown(
        """
        <style>
            .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
            [data-testid="stMetric"] {
                background: rgba(35, 45, 60, 0.38);
                border: 1px solid rgba(128, 145, 165, 0.24);
                border-radius: 0.75rem;
                padding: 1rem 1.1rem;
            }
            [data-testid="stMetricLabel"] {font-weight: 600;}
            div[data-testid="stDownloadButton"] button {width: 100%;}
            .dashboard-subtitle {color: #8d99a8; margin: -0.6rem 0 1.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_line_figure(
    dataframes: list[pd.DataFrame],
    title: str,
    y_label: str,
    line_styles: tuple[str, ...] | None = None,
) -> plt.Figure:
    """Create a consistently formatted line chart from one or more frames."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    styles = line_styles or tuple("-" for _ in dataframes)

    for dataframe, style in zip(dataframes, styles):
        for column in dataframe.columns:
            ax.plot(
                dataframe.index,
                dataframe[column].values,
                linestyle=style,
                linewidth=2,
                label=column,
            )

    ax.set_title(title, fontsize=16, fontweight="semibold", pad=16)
    ax.set_xlabel("Sols since landing", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def display_figure(figure: plt.Figure) -> None:
    """Render and close a Matplotlib figure to avoid retaining UI figures."""
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    """Serialize a result frame for a Streamlit download button."""
    return dataframe.to_csv(index=True).encode("utf-8")


def build_summary_table(results: dict) -> pd.DataFrame:
    """Build a compact, presentation-only summary of calculated arrays."""
    rows = []
    for site in results["selected_sites"]:
        key = site["site_name"]
        rows.append(
            {
                "Landing site": site["full_name"].strip(),
                "Avg energy 2028 (kWh/m²)": results["energy_2028"][key].iloc[-1],
                "Avg energy 2031 (kWh/m²)": results["energy_2031"][key].iloc[-1],
                "Avg payload 2028 (h)": results["payload_2028"][key].mean(),
                "Avg payload 2031 (h)": results["payload_2031"][key].mean(),
                "Worst DoD 2028 (%)": results["battery_2028"][key].max(),
                "Worst DoD 2031 (%)": results["battery_2031"][key].max(),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Verified scientific workflow orchestration
# -----------------------------------------------------------------------------

def run_analysis(selected_sites: list[dict], config: dict, progress_bar) -> dict:
    """Run the original calculation sequence and return display-ready frames."""
    energy_raw = pd.DataFrame()
    payload_2028 = pd.DataFrame(index=SUPPORT_SOLS)
    payload_2031 = pd.DataFrame(index=SUPPORT_SOLS)
    battery_2028 = pd.DataFrame(index=SUPPORT_SOLS)
    battery_2031 = pd.DataFrame(index=SUPPORT_SOLS)

    total_sites = len(selected_sites)
    for site_index, site in enumerate(selected_sites):
        progress_bar.progress(
            site_index / total_sites,
            text=f"Analyzing {site['full_name'].strip()}...",
        )

        # Atmospheric environment and radiative transfer: unchanged call sequence.
        _, tau_lc, _, tau_lcs, g_c = water_ice_cloud()
        albedo = surface_albedo()
        tau_ga, tau_gs = atm_gas()
        g_d, w_d, q_d = optical_params()
        tau_rd = get_optical_depth(site=site)
        tau_ld, tau_lds, _ = all_wavelength_dust_od(tau_rd, q_d, w_d)
        tau_total = tau_lc + tau_ga + tau_gs + tau_ld
        w_total = (tau_lcs + tau_lds + tau_gs) / tau_total
        g_total = (tau_lds * g_d + tau_lcs * g_c) / (tau_lcs + tau_lds + tau_gs)
        w_total[w_total > 1] = 1

        mars_spectrum = am_mars_spectrum()
        (
            effective_total,
            _,
            beam_total,
            diffuse_total,
            _,
            _,
            beam_daily,
            diffuse_daily,
            toa_maximum,
            _,
            beam_maximum,
            diffuse_maximum,
        ) = complete_year_irradiance(
            lat=site["lat"],
            tau_total=tau_total,
            w_total=w_total,
            g_total=g_total,
            A_l=albedo,
            E_ml=mars_spectrum,
        )
        effective_total = beam_total + 0.86 * diffuse_total
        effective_maximum = beam_maximum + 0.86 * diffuse_maximum
        effective_daily = beam_daily + 0.86 * diffuse_daily

        # Preserve the backend's established plot and CSV artifacts.
        plot_daily_maxima(
            site_name=site["full_name"],
            Ls=LS,
            Ew1=toa_maximum,
            Ef_w1=effective_maximum,
            Bw1=beam_maximum,
            Dw1=diffuse_maximum,
        )
        plot_available_energy_wo_dust(
            site_name=site["full_name"], Ls=LS, Ef_p=effective_daily
        )
        export_complete_year_irradiance(
            site_name=site["site_name"],
            Ls=LS,
            Ew1=toa_maximum,
            Ef_w1=effective_maximum,
            Bw1=beam_maximum,
            Dw1=diffuse_maximum,
        )

        dust_reference = get_dd_ref()
        dust_2028, solar_2028, eclipse_2028 = dd_fac_2028(
            site=site,
            ref_dd_mean=dust_reference,
            Ef_p=effective_daily,
            Ef_t=effective_total,
        )
        site_energy_2028 = avail_energy_2028(
            sol_pow=solar_2028, new=dust_2028, site_name=site["full_name"]
        )
        dust_2031, solar_2031, eclipse_2031 = dd_fac_2031(
            site=site,
            ref_dd_mean=dust_reference,
            Ef_p=effective_daily,
            Ef_t=effective_total,
        )
        site_energy_2031 = avail_energy_2031(
            sol_pow=solar_2031, new=dust_2031, site_name=site["full_name"]
        )

        key = site["site_name"]
        energy_raw[f"{key}2028"] = site_energy_2028
        energy_raw[f"{key}2031"] = site_energy_2031
        payload_2028[key], battery_2028[key] = load_support_duration(
            ecl_dur=np.array(eclipse_2028),
            sol_pow_smooth=site_energy_2028,
            ecl_load=config["eclipse_load"],
            sunlit_load=config["sunlit_load"],
            payload=config["payload_power"],
            bat_size=config["battery_size"],
            disch_vlt=config["discharge_voltage"],
            bat_eff=BATTERY_EFFICIENCY,
            panel_size=config["panel_size"],
            panel_eff=PANEL_EFFICIENCY,
        )
        payload_2031[key], battery_2031[key] = load_support_duration(
            ecl_dur=np.array(eclipse_2031),
            sol_pow_smooth=site_energy_2031,
            ecl_load=config["eclipse_load"],
            sunlit_load=config["sunlit_load"],
            payload=config["payload_power"],
            bat_size=config["battery_size"],
            disch_vlt=config["discharge_voltage"],
            bat_eff=BATTERY_EFFICIENCY,
            panel_size=config["panel_size"],
            panel_eff=PANEL_EFFICIENCY,
        )
        plt.close("all")

    progress_bar.progress(1.0, text="Analysis complete")

    cumulative_mean = energy_raw.cumsum(axis=0).divide(
        np.arange(1, len(energy_raw) + 1), axis=0
    )
    cumulative_mean.index = MISSION_SOLS
    energy_2028 = cumulative_mean.filter(regex="2028$").rename(
        columns=lambda name: name[:-4]
    )
    energy_2031 = cumulative_mean.filter(regex="2031$").rename(
        columns=lambda name: name[:-4]
    )
    return {
        "selected_sites": selected_sites,
        "config": config,
        "energy_2028": energy_2028,
        "energy_2031": energy_2031,
        "payload_2028": payload_2028,
        "payload_2031": payload_2031,
        "battery_2028": battery_2028,
        "battery_2031": battery_2031,
    }


# -----------------------------------------------------------------------------
# Dashboard sections
# -----------------------------------------------------------------------------

def render_kpis(results: dict) -> None:
    """Render the four requested mission KPIs."""
    all_energy = pd.concat([results["energy_2028"], results["energy_2031"]], axis=1)
    all_payload = pd.concat([results["payload_2028"], results["payload_2031"]], axis=1)
    all_battery = pd.concat([results["battery_2028"], results["battery_2031"]], axis=1)

    columns = st.columns(4)
    columns[0].metric("Selected landing sites", len(results["selected_sites"]))
    columns[1].metric("Average available energy", f"{all_energy.iloc[-1].mean():.2f} kWh/m²")
    columns[2].metric("Average payload duration", f"{all_payload.to_numpy().mean():.2f} h")
    columns[3].metric("Worst battery DoD", f"{all_battery.to_numpy().max():.2f}%")


def render_overview(results: dict) -> None:
    st.subheader("Mission performance summary")
    st.caption("Summary statistics include both the 2028 and 2031 launch windows.")
    summary = build_summary_table(results)
    st.dataframe(
        summary.style.format(precision=2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Analysis configuration")
    config = results["config"]
    st.markdown(
        f"""
        **Power profile:** {config['eclipse_load']} W nighttime load ·
        {config['sunlit_load']} W daytime load · {config['payload_power']} W payload

        **Spacecraft:** {config['panel_size']:.1f} m² solar array ·
        {config['battery_size']} Ah battery · {config['discharge_voltage']} V discharge voltage
        """
    )


def render_energy(results: dict) -> None:
    display_figure(
        create_line_figure(
            [results["energy_2028"], results["energy_2031"]],
            "Cumulative Mean Available Energy",
            "Cumulative mean energy (kWh/m²)",
            ("-", "--"),
        )
    )
    st.caption("Solid lines: 2028 launch window · Dashed lines: 2031 launch window")


def render_window_charts(results: dict, metric: str, y_label: str, title: str) -> None:
    columns = st.columns(2)
    for column, year in zip(columns, (2028, 2031)):
        with column:
            display_figure(
                create_line_figure(
                    [results[f"{metric}_{year}"]],
                    f"{title} · {year} Launch Window",
                    y_label,
                )
            )


def render_downloads(results: dict) -> None:
    st.subheader("Generated analysis data")
    st.caption("Download the calculated comparison series currently shown in the dashboard.")
    downloads = [
        ("Energy · 2028", "energy_2028", "comparison_energy_2028.csv"),
        ("Energy · 2031", "energy_2031", "comparison_energy_2031.csv"),
        ("Payload · 2028", "payload_2028", "comparison_load_support_2028.csv"),
        ("Payload · 2031", "payload_2031", "comparison_load_support_2031.csv"),
        ("Battery DoD · 2028", "battery_2028", "comparison_battery_dod_2028.csv"),
        ("Battery DoD · 2031", "battery_2031", "comparison_battery_dod_2031.csv"),
    ]
    for row_start in range(0, len(downloads), 2):
        columns = st.columns(2)
        for column, (label, key, filename) in zip(columns, downloads[row_start : row_start + 2]):
            with column:
                st.download_button(
                    label=f"Download {label} CSV",
                    data=dataframe_to_csv(results[key]),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{key}",
                )


# -----------------------------------------------------------------------------
# Streamlit application
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Mars Mission Power Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_page_style()

st.title("Mars Mission Power Analysis")
st.markdown(
    '<p class="dashboard-subtitle">Physics-based assessment of surface energy, payload support, and battery depth of discharge.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analysis Controls")
    st.caption("Configure the mission, then run the full scientific workflow.")

    with st.form("mission_configuration"):
        st.subheader("Mission Configuration")
        selected_site_names = st.multiselect(
            "Landing sites",
            [site["full_name"] for site in SITES],
            default=[site["full_name"] for site in SITES[:2]],
        )

        st.subheader("Power Profile")
        eclipse_load = st.slider("Nighttime load (W)", 100, 250, 120, 10)
        sunlit_load = st.slider("Daytime load (W)", 100, 250, 200, 10)
        payload_power = st.slider("Payload requirement (W)", 100, 200, 150, 5)

        st.subheader("Spacecraft Configuration")
        panel_size = st.slider("Solar panel area (m²)", 4.0, 8.0, 7.0, 0.5)
        battery_size = st.slider("Battery capacity (Ah)", 100, 500, 250, 50)
        discharge_voltage = st.slider("Battery discharge voltage (V)", 20, 40, 25, 5)
        st.caption(
            f"Fixed efficiencies · Panel {PANEL_EFFICIENCY:.0%} · Battery {BATTERY_EFFICIENCY:.0%}"
        )

        run_requested = st.form_submit_button(
            "Run Analysis", type="primary", use_container_width=True
        )

if run_requested:
    selected_sites = [site for site in SITES if site["full_name"] in selected_site_names]
    if not selected_sites:
        st.warning("Select at least one landing site before running the analysis.")
    else:
        configuration = {
            "eclipse_load": eclipse_load,
            "sunlit_load": sunlit_load,
            "payload_power": payload_power,
            "panel_size": panel_size,
            "battery_size": battery_size,
            "discharge_voltage": discharge_voltage,
        }
        progress = st.progress(0.0, text="Preparing analysis...")
        with st.spinner("Running radiative transfer and mission power calculations..."):
            try:
                st.session_state["analysis_results"] = run_analysis(
                    selected_sites, configuration, progress
                )
            except Exception as error:
                st.session_state.pop("analysis_results", None)
                st.error(f"Analysis could not be completed: {error}")
            else:
                st.success("Analysis complete. Results are ready below.")
        progress.empty()

results = st.session_state.get("analysis_results")
if results is None:
    st.info("Choose one or more landing sites in the sidebar and select **Run Analysis**.")
    st.stop()

render_kpis(results)
st.divider()

overview_tab, energy_tab, payload_tab, battery_tab, downloads_tab = st.tabs(
    ["Overview", "Energy", "Payload", "Battery", "Downloads"]
)
with overview_tab:
    render_overview(results)
with energy_tab:
    render_energy(results)
with payload_tab:
    render_window_charts(results, "payload", "Payload operation duration (hours)", "Payload Support")
with battery_tab:
    render_window_charts(results, "battery", "Depth of discharge (%)", "Battery Depth of Discharge")
with downloads_tab:
    render_downloads(results)
