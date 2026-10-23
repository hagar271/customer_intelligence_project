"""Customer Intelligence System - Streamlit dashboard.

If generated model/result artifacts are absent, show a clearly labelled demo preview
so the app remains presentable without the original dataset.
"""
import streamlit as st

st.set_page_config(page_title="Customer Intelligence System", page_icon="📊", layout="wide")

import config  # noqa: E402
import data_loader  # noqa: E402
from utils import inject_css  # noqa: E402
import demo_dashboard  # noqa: E402
import page_churn as churn  # noqa: E402
import page_overview as overview  # noqa: E402
import page_pipeline_runner as pipeline_runner  # noqa: E402
import page_prediction as prediction  # noqa: E402
import page_sales as sales  # noqa: E402
import page_segmentation as segmentation  # noqa: E402

PAGES = {
    "📊 Overview": overview.render,
    "🛒 Sales Analysis": sales.render,
    "🧩 Customer Segmentation": segmentation.render,
    "🔮 Churn Prediction": churn.render,
    "🎯 Customer Prediction": prediction.render,
    "⚙️ Pipeline & Deployment": pipeline_runner.render,
}


def sidebar():
    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Online Retail II")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()), index=0, key="nav",
                            label_visibility="collapsed")
    with st.sidebar.expander("System check (files & versions)"):
        st.dataframe(data_loader.artifact_status(), hide_index=True)
        if data_loader.exists("churn_meta"):
            try:
                import sklearn
                trained_with = data_loader.load_meta().get("sklearn_version")
                if trained_with and trained_with != sklearn.__version__:
                    st.warning(
                        f"Model trained with scikit-learn {trained_with}, but this app runs "
                        f"{sklearn.__version__}. Install the same version to avoid loading errors."
                    )
            except Exception:
                pass
    return page


def main():
    inject_css()
    missing = data_loader.missing_required()

    # When the repo has source code but no generated artifacts, don't open a broken
    # page or force a dataset upload. Show the presentation preview by default.
    if missing:
        st.sidebar.title("Customer Intelligence")
        st.sidebar.caption("Online Retail II")
        page = st.sidebar.radio(
            "Navigate",
            ["📊 Dashboard Preview", "⚙️ Pipeline & Deployment"],
            index=0,
            key="demo_nav",
            label_visibility="collapsed",
        )
        with st.sidebar.expander("System check (files & versions)"):
            st.dataframe(data_loader.artifact_status(), hide_index=True)
        if page == "📊 Dashboard Preview":
            demo_dashboard.render()
        else:
            pipeline_runner.render()
        return

    page = sidebar()
    try:
        PAGES[page]()
    except data_loader.ArtifactError as e:
        st.error(str(e))
        st.info("Open **Pipeline & Deployment** to review the required generated files.")
    except Exception as e:
        st.error(f"Something went wrong while building this page: {e}")
        with st.expander("Technical details"):
            st.exception(e)


main()
