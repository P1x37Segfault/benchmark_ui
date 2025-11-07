import os
import io
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
import streamlit as st

Z_95 = 1.96  # 95% quantile for standard normal distribution

category_colors = {
    "Reference": "#1f77b4",
    "ROCm": "#d62728",
    "Kokkos": "#ff7f0e",
    "OpenMP": "#2ca02c",
    "other": "#9467bd",
}

plt.rcParams.update(
    {
        "font.size": 16,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (13, 7),  # Double column width
        "lines.linewidth": 2,
        "axes.linewidth": 1.5,
        "grid.linewidth": 0.5,
    }
)

def name_to_color(name):
    cat = get_category(name)
    base = category_colors.get(cat, category_colors["other"])
    return base


def name_to_display(name):
    return True


def format_benchmark_name(name):
    if "_" in name:
        parts = name.split("_")
        formatted = [part.capitalize() for part in parts]
        return " ".join(formatted)
    else:
        return name.upper()


def get_category(name):
    name_lower = name.lower()
    if "reference" in name_lower:
        return "Reference"
    elif "rocm" in name_lower:
        return "ROCm"
    elif "kokkos" in name_lower:
        return "Kokkos"
    elif "openmp" in name_lower:
        return "OpenMP"
    else:
        return "other"


def name_to_dotted_line(name):
    return False


def plot_default(name: str, output_name: str = ""):
    csv_path = os.path.join("results", f"{name}.csv")
    output_path = os.path.join("results", f"{output_name or name}.pdf")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    # Filter out entries where correctness fields are 0 (false)
    correct_fields = ["correctresult", "correctminvalue", "correctminindex"]
    for field in correct_fields:
        if field in df.columns:
            df = df[df[field] != 0]

    col_name = "name"
    col_points = "points"
    col_time = "time"

    names = df[col_name].unique()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])

    for name in names:
        if not name_to_display(name):
            continue

        sub = df[df[col_name] == name]
        points = sorted(sub[col_points])
        grouped = sub.groupby(col_points)

        # Gather mean, std and sample size for every x-value
        mean_time = []
        std_time = []
        n_obs = []
        for p in points:
            g = grouped.get_group(p)
            mean_time.append(g[col_time].mean())
            std_time.append(g[col_time].std(ddof=1))
            n_obs.append(len(g))

        mean_time = np.array(mean_time)
        std_time = np.array(std_time)
        n_obs = np.array(n_obs)

        # standard error
        se = std_time / np.sqrt(n_obs)

        # 95% pointwise CI for the mean
        lower = mean_time - Z_95 * se
        upper = mean_time + Z_95 * se

        ax1.plot(
            points,
            mean_time,
            label=name,
            color=name_to_color(name),
            linestyle="--" if name_to_dotted_line(name) else "-",
        )
        ax1.fill_between(
            points,
            lower,
            upper,
            color=name_to_color(name),
            alpha=0.3,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_locator(LogLocator(base=10.0))
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Points", fontsize=16)
    ax1.set_ylabel("Seconds", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_data():
    data = []
    for subdir in ["FTP", "UC3"]:
        path = os.path.join("results", subdir)
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".csv"):
                benchmark = file[:-4]  # remove .csv
                df = pd.read_csv(os.path.join(path, file))
                df.columns = [c.lower() for c in df.columns]
                df["source"] = subdir
                df["benchmark"] = benchmark
                data.append(df)
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        return pd.DataFrame()


def create_gui(df):
    st.set_page_config(layout="wide", page_title="Benchmark UI")

    # Benchmark selection via tabs
    benchmarks = sorted(df["benchmark"].unique()) if not df.empty else []
    if not benchmarks:
        st.write("No benchmarks found")
        return

    tabs = st.tabs([format_benchmark_name(b) for b in benchmarks])

    # Get all unique names across all benchmarks
    all_unique_names = df["name"].unique()

    with st.sidebar:
        st.title("Benchmark UI")
        st.header("Filters")
        if st.button(
            "Reset all colors", on_click=reset_all_colors, args=(all_unique_names,)
        ):
            pass  # The action is in on_click
        # Sources
        with st.expander("Sources", expanded=True):
            sources = ["FTP", "UC3"]
            source_vars = {s: st.checkbox(f"{s}", value=True) for s in sources}

        # Get present categories across all
        present_categories = set(get_category(name) for name in all_unique_names)
        categories = [
            cat
            for cat in ["Kokkos", "OpenMP", "ROCm", "Reference"]
            if cat in present_categories
        ]

        # Group individual names by category
        name_checks = {}
        user_colors = {}
        disable_all = {}
        for cat in categories:
            with st.expander(cat, expanded=True):
                cat_names = [
                    name
                    for name in sorted(all_unique_names)
                    if get_category(name) == cat
                ]
                if len(cat_names) > 2:
                    disable_all[cat] = st.checkbox(
                        "Disable all", value=False, key=f"disable_{cat}"
                    )
                for name in cat_names:
                    col1, col2, col3 = st.columns(
                        [3, 1, 1], vertical_alignment="center"
                    )
                    with col1:
                        if cat in disable_all and disable_all[cat]:
                            st.checkbox(
                                name,
                                value=st.session_state.get(name, True),
                                disabled=True,
                                key=f"disabled_{name}",
                            )
                        else:
                            name_checks[name] = st.checkbox(
                                name, value=st.session_state.get(name, True), key=name
                            )
                    color_key = f"color_{name}"
                    with col2:
                        user_colors[name] = st.color_picker(
                            f"label_for_color_{name}",
                            value=st.session_state.get(color_key, name_to_color(name)),
                            key=color_key,
                            label_visibility="collapsed",
                        )
                    with col3:
                        if st.session_state.get(
                            color_key, name_to_color(name)
                        ) != name_to_color(name):
                            st.button(
                                "",
                                key=f"reset_{name}",
                                on_click=reset_color,
                                args=(name,),
                                icon="❌",
                            )

    for i, benchmark in enumerate(benchmarks):
        with tabs[i]:
            # Get unique names for the selected benchmark
            unique_names = df[df["benchmark"] == benchmark]["name"].unique()

            # Filter df
            filtered_df = df[df["benchmark"] == benchmark]
            filtered_df = filtered_df[
                filtered_df["source"].isin([s for s in sources if source_vars[s]])
            ]

            # Apply category and individual name filters
            selected_names = set()
            for cat in categories:
                if not (cat in disable_all and disable_all[cat]):
                    selected_names.update(
                        [
                            name
                            for name in unique_names
                            if get_category(name) == cat
                            and name in name_checks
                            and name_checks[name]
                        ]
                    )
            filtered_df = filtered_df[filtered_df["name"].isin(selected_names)]

            if filtered_df.empty:
                st.write("No data to plot")
            else:
                plot_placeholder = st.empty()
                current_filters = (
                    tuple(sorted(source_vars.items())),
                    tuple(sorted(disable_all.items())),
                    tuple(sorted(name_checks.items())),
                )
                benchmark_key = f"last_fig_{benchmark}"
                filter_key = f"last_filters_{benchmark}"
                regenerating = (
                    filter_key not in st.session_state
                    or st.session_state[filter_key] != current_filters
                )
                if regenerating and benchmark_key in st.session_state:
                    plot_placeholder.pyplot(st.session_state[benchmark_key])

                fig = generate_plot(filtered_df, user_colors)
                plot_placeholder.pyplot(fig)
                # Generate PDF for download
                buf = io.BytesIO()
                fig.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                pdf_bytes = buf.getvalue()
                plt.close(fig)
                st.session_state[benchmark_key] = fig
                st.session_state[filter_key] = current_filters
                
                # View PDF link
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                link = f'<a href="data:application/pdf;base64,{pdf_base64}" target="_blank">View PDF</a>'
                st.markdown(link, unsafe_allow_html=True)


@st.cache_data
def generate_plot(filtered_df, user_colors):
    # Filter out entries where correctness fields are 0 (false)
    correct_fields = ["correctresult", "correctminvalue", "correctminindex"]
    for field in correct_fields:
        if field in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[field] != 0]

    col_name = "name"
    col_points = "points"
    col_time = "time"

    names = filtered_df[col_name].unique()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])

    for name in names:
        if not name_to_display(name):
            continue

        cat = get_category(name)
        cat_names_in_plot = [n for n in names if get_category(n) == cat]
        if len(cat_names_in_plot) >= 2:
            index = cat_names_in_plot.index(name)
            linestyle = "--" if index % 2 == 0 else "-"
        else:
            linestyle = "-"

        sub = filtered_df[filtered_df[col_name] == name]
        points = sorted(sub[col_points])
        grouped = sub.groupby(col_points)

        # Gather mean, std and sample size for every x-value
        mean_time = []
        std_time = []
        n_obs = []
        for p in points:
            g = grouped.get_group(p)
            mean_time.append(g[col_time].mean())
            std_time.append(g[col_time].std(ddof=1))
            n_obs.append(len(g))

        mean_time = np.array(mean_time)
        std_time = np.array(std_time)
        n_obs = np.array(n_obs)

        # standard error
        se = std_time / np.sqrt(n_obs)

        # 95% pointwise CI for the mean
        lower = mean_time - Z_95 * se
        upper = mean_time + Z_95 * se

        ax1.plot(
            points,
            mean_time,
            label=name,
            color=user_colors[name],
            linestyle=linestyle,
        )
        ax1.fill_between(
            points,
            lower,
            upper,
            color=user_colors[name],
            alpha=0.3,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_locator(LogLocator(base=10.0))
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Points", fontsize=16)
    ax1.set_ylabel("Seconds", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=16)

    plt.tight_layout()

    return fig


def reset_color(name):
    st.session_state[f"color_{name}"] = name_to_color(name)


def reset_all_colors(all_unique_names):
    for name in all_unique_names:
        st.session_state[f"color_{name}"] = name_to_color(name)


if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("No data found")
    else:
        create_gui(df)
