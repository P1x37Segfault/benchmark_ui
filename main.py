import os
import io
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
import matplotlib.ticker as ticker
import streamlit as st

Z_95 = 1.96  # 95% quantile for standard normal distribution

category_colors = {
    "Reference": "#1f77b4",
    "ROCm": "#d62728",
    "ROCm + OpenMP": "#d62728",
    "Kokkos": "#ff7f0e",
    "OpenMP": "#2ca02c",
    "default": "#000000",
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


def get_display_name(name):
    if "(" in name and ")" in name:
        start = name.index("(")
        end = name.index(")")
        return name[start + 1 : end].strip()
    return "default"


def name_to_color(name):
    cat = get_category(name)
    c = category_colors.get(cat, category_colors["default"])
    return c


def format_benchmark_name(name):
    if "_" in name:
        parts = name.split("_")
        formatted = [part.capitalize() for part in parts]
        return " ".join(formatted)
    else:
        return name.capitalize()


def get_category(name):
    name_lower = name.lower()
    if "reference" in name_lower:
        return "Reference"
    elif "rocm + openmp" in name_lower:
        return "ROCm + OpenMP"
    elif "rocm" in name_lower:
        return "ROCm"
    elif "kokkos" in name_lower:
        return "Kokkos"
    elif "openmp" in name_lower:
        return "OpenMP"
    else:
        return "other"


def load_data():
    data = []
    results_path = "results"
    if os.path.exists(results_path):
        for subdir in os.listdir(results_path):
            path = os.path.join(results_path, subdir)
            if os.path.isdir(path):
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
            for cat in ["Kokkos", "OpenMP", "ROCm + OpenMP", "ROCm", "Reference"]
            if cat in present_categories
        ]

        # Group individual names by category
        name_checks = {}
        user_colors = {}
        user_linestyles = {}
        disable_all = {}
        for cat in categories:
            with st.expander(cat, expanded=True):
                cat_names = [
                    name
                    for name in sorted(all_unique_names)
                    if get_category(name) == cat
                ]
                if len(cat_names) >= 2:
                    disable_all[cat] = st.checkbox(
                        "Disable all", value=False, key=f"disable_{cat}"
                    )
                for name in cat_names:
                    col1, col2, col3 = st.columns(
                        [2, 1.5, 2],
                        gap="small",
                        vertical_alignment="center",
                        width="stretch",
                    )
                    with col1:
                        if cat in disable_all and disable_all[cat]:
                            st.checkbox(
                                get_display_name(name),
                                value=st.session_state.get(name, True),
                                disabled=True,
                                key=f"disabled_{name}",
                            )
                        else:
                            name_checks[name] = st.checkbox(
                                get_display_name(name),
                                value=st.session_state.get(name, True),
                                key=name,
                            )
                    user_linestyles[name] = st.session_state.get(
                        f"linestyle_{name}", "-"
                    )
                    with col2:
                        current_style = st.session_state.get(f"linestyle_{name}", "-")
                        style_label = {"-": "line", "--": "dash", ":": "dot"}[
                            current_style
                        ]
                        st.button(
                            style_label,
                            key=f"style_{name}",
                            on_click=cycle_linestyle,
                            args=(name,),
                            width="stretch",
                            use_container_width=True,
                        )
                    color_key = f"color_{name}"
                    with col3:
                        col31, col32 = st.columns(
                            [1, 1],
                            gap="small",
                            vertical_alignment="center",
                            width="stretch",
                        )
                        with col31:
                            user_colors[name] = st.color_picker(
                                f"label_for_color_{name}",
                                value=st.session_state.get(
                                    color_key, name_to_color(name)
                                ),
                                key=color_key,
                                label_visibility="collapsed",
                            )
                        with col32:
                            if st.session_state.get(
                                color_key, name_to_color(name)
                            ) != name_to_color(name):
                                st.button(
                                    "X",
                                    key=f"reset_{name}",
                                    on_click=reset_color,
                                    args=(name,),
                                    # icon="❌",
                                )
                    # with col4:

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
                    tuple(sorted(user_linestyles.items())),
                )
                benchmark_key = f"last_fig_{benchmark}"
                filter_key = f"last_filters_{benchmark}"
                regenerating = (
                    filter_key not in st.session_state
                    or st.session_state[filter_key] != current_filters
                )
                if regenerating and benchmark_key in st.session_state:
                    plot_placeholder.pyplot(st.session_state[benchmark_key])

                fig = generate_plot(filtered_df, user_colors, user_linestyles)
                plot_placeholder.pyplot(fig)
                # Generate PDF for download
                buf = io.BytesIO()
                fig.savefig(buf, format="pdf", bbox_inches="tight")
                buf.seek(0)
                pdf_bytes = buf.getvalue()
                plt.close(fig)
                st.session_state[benchmark_key] = fig
                st.session_state[filter_key] = current_filters

                # View PDF link
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                link = f'<a href="data:application/pdf;base64,{pdf_base64}" target="_blank">View PDF</a>'
                st.markdown(link, unsafe_allow_html=True)

                # error plot for monte carlo
                if benchmark == "monte_carlo":
                    error_fig = generate_error_plot(
                        filtered_df, user_colors, user_linestyles
                    )
                    st.pyplot(error_fig)
                    # Generate PDF for error plot
                    error_buf = io.BytesIO()
                    error_fig.savefig(error_buf, format="pdf", bbox_inches="tight")
                    error_buf.seek(0)
                    error_pdf_bytes = error_buf.getvalue()
                    plt.close(error_fig)
                    error_pdf_base64 = base64.b64encode(error_pdf_bytes).decode("utf-8")
                    error_link = f'<a href="data:application/pdf;base64,{error_pdf_base64}" target="_blank">View Error PDF</a>'
                    st.markdown(error_link, unsafe_allow_html=True)

                # speedup plot
                unique_points = sorted(filtered_df["points"].unique())
                if unique_points:
                    point_options = {f"10^{int(np.log10(p))}": p for p in unique_points}
                    selected_label = st.selectbox(
                        "Select num_points for speedup comparison",
                        list(point_options.keys()),
                        key=f"points_{benchmark}",
                    )
                    selected_points = point_options[selected_label]
                    use_decimals = st.checkbox("Use decimals in speedup y-axis", value=False, key=f"decimals_{benchmark}")
                    speedup_fig = generate_speedup_plot(
                        filtered_df, user_colors, user_linestyles, selected_points, use_decimals
                    )

                    if speedup_fig is None:
                        st.write(
                            f"Error generating speedup plot for {selected_points} points."
                        )
                    else:
                        st.pyplot(speedup_fig)
                        # Generate PDF for speedup plot
                        speedup_buf = io.BytesIO()
                        speedup_fig.savefig(speedup_buf, format="pdf", bbox_inches="tight")
                        speedup_buf.seek(0)
                        speedup_pdf_bytes = speedup_buf.getvalue()
                        plt.close(speedup_fig)
                        speedup_pdf_base64 = base64.b64encode(speedup_pdf_bytes).decode(
                            "utf-8"
                        )
                        speedup_link = f'<a href="data:application/pdf;base64,{speedup_pdf_base64}" target="_blank">View Speedup PDF</a>'
                        st.markdown(speedup_link, unsafe_allow_html=True)


@st.cache_data
def generate_plot(filtered_df, user_colors, user_linestyles):
    col_name = "name"
    col_points = "points"
    col_time = "time"

    names = filtered_df[col_name].unique()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])

    for name in names:
        linestyle = user_linestyles.get(name, "-")

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
    ax1.grid(True, which="major", axis="both", linestyle="--", alpha=0.7, linewidth=1.0)
    ax1.set_xlabel("Points", fontsize=16)
    ax1.set_ylabel("Seconds", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.legend(fontsize=16)

    plt.tight_layout()

    return fig


@st.cache_data
def generate_error_plot(filtered_df, user_colors, user_linestyles):
    col_name = "name"
    col_points = "points"
    col_error = "error"

    names = filtered_df[col_name].unique()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])

    for name in names:
        linestyle = user_linestyles.get(name, "-")

        sub = filtered_df[filtered_df[col_name] == name]
        points = sorted(sub[col_points])
        grouped = sub.groupby(col_points)

        # Gather mean, std and sample size for every x-value
        mean_error = []
        std_error = []
        n_obs = []
        for p in points:
            g = grouped.get_group(p)
            mean_error.append(g[col_error].mean())
            std_error.append(g[col_error].std(ddof=1))
            n_obs.append(len(g))

        mean_error = np.array(mean_error)
        std_error = np.array(std_error)
        n_obs = np.array(n_obs)

        # standard error
        se = std_error / np.sqrt(n_obs)

        # no confidence bands here - too messy
        # 95% pointwise CI for the mean
        # lower = mean_error - Z_95 * se
        # upper = mean_error + Z_95 * se

        ax1.plot(
            points,
            mean_error,
            label=name,
            color=user_colors[name],
            linestyle=linestyle,
        )

    all_points = sorted(df[col_points].unique())
    ax1.plot(
        all_points,
        1 / np.sqrt(np.array(all_points)),
        linestyle=":",
        color="black",
        label=r"$\frac{1}{\sqrt{N}}$",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_locator(LogLocator(base=10.0))
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0))
    ax1.grid(True, which="major", axis="both", linestyle="--", alpha=0.7, linewidth=1.0)
    ax1.set_xlabel("Points", fontsize=16)
    ax1.set_ylabel("Error", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.legend(fontsize=16)

    plt.tight_layout()

    return fig


def generate_speedup_plot(filtered_df, user_colors, user_linestyles, selected_points, use_decimals):
    # Find reference data for selected points
    ref_df = filtered_df[
        (filtered_df["name"].str.lower().str.contains("reference"))
        & (filtered_df["points"] == selected_points)
    ]
    if ref_df.empty:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            f"No reference data found for {selected_points} points",
            transform=ax.transAxes,
            ha="center",
        )
        return fig

    ref_mean = ref_df["time"].mean()

    if ref_mean == 0:
        # error
        return

    # Get all implementation names except reference
    all_names = filtered_df["name"].unique()
    impl_names = sorted([name for name in all_names if "reference" not in name.lower()])

    speedups = []
    labels = []
    colors = []
    hatches = []
    for name in impl_names:
        impl_df = filtered_df[
            (filtered_df["name"] == name) & (filtered_df["points"] == selected_points)
        ]
        if not impl_df.empty:
            impl_mean = impl_df["time"].mean()

            if impl_mean == 0:
                # unexpected, skip
                continue
            elif impl_mean == ref_mean:
                speedup = 1.0
            else:
                speedup = ref_mean / impl_mean
            speedups.append(speedup)
            labels.append(name)
            colors.append(user_colors.get(name, "#000000"))
            linestyle = user_linestyles.get(name, "-")
            hatch_map = {"-": "", "--": "/", ":": "\\"}
            hatches.append(hatch_map.get(linestyle, ""))

    if not labels:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            f"No implementation data found for {selected_points} points",
            transform=ax.transAxes,
            ha="center",
        )
        return fig

    # Sort by speedup ascending
    sorted_indices = sorted(range(len(speedups)), key=lambda i: speedups[i])
    speedups = [speedups[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    hatches = [hatches[i] for i in sorted_indices]

    fmt_str = ".1f" if use_decimals else ".0f"

    # Adjust figure width based on number of bars
    bar_width = 0.6
    fig_width = max(6, len(labels) * 0.8)  # Minimum 6 inches, expand for more bars
    fig_height = 6
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, speedups, color=colors, hatch=hatches, width=bar_width)

    # Apply hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Add speedup labels on top of bars
    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(speedups) * 0.02,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_ylim(0, max(speedups) * 1.1)
    ax.set_ylabel("Speedup", fontsize=16)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:{fmt_str}}x"))
    ax.set_title(f"10^{int(np.log10(selected_points))} points", fontsize=16)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.7, linewidth=1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def reset_color(name):
    st.session_state[f"color_{name}"] = name_to_color(name)


def reset_all_colors(all_unique_names):
    for name in all_unique_names:
        st.session_state[f"color_{name}"] = name_to_color(name)


def cycle_linestyle(name):
    current = st.session_state.get(f"linestyle_{name}", "-")
    styles = ["-", "--", ":"]
    next_index = (styles.index(current) + 1) % len(styles)
    st.session_state[f"linestyle_{name}"] = styles[next_index]


if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("No data found")
    else:
        create_gui(df)
