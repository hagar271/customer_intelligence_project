"""Customer Intelligence System - Streamlit dashboard.

Run with:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Customer Intelligence System", page_icon="📊", layout="wide")

import config  # noqa: E402  (imports must come after set_page_config)
import data_loader  # noqa: E402
from utils import inject_css  # noqa: E402
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
    "⚙️ Run Pipeline": pipeline_runner.render,
}


def sidebar():
    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Online Retail II")
    # First run (required files missing) -> open the "Run Pipeline" page by default
    missing = data_loader.missing_required()
    default = list(PAGES).index("⚙️ Run Pipeline") if missing else 0
    page = st.sidebar.radio("Navigate", list(PAGES.keys()), index=default, key="nav", label_visibility="collapsed")

    with st.sidebar.expander("System check (files & versions)"):
        st.dataframe(data_loader.artifact_status(), hide_index=True)
        if data_loader.exists("churn_meta"):
            try:
                import sklearn
                trained_with = data_loader.load_meta().get("sklearn_version")
                if trained_with and trained_with != sklearn.__version__:
                    st.warning(f"Model trained with scikit-learn {trained_with}, but this app runs {sklearn.__version__}. "
                               "Install the same version to avoid loading errors.")
            except Exception:
                pass
    return page


def main():
    inject_css()
    page = sidebar()
    try:
        PAGES[page]()
    except data_loader.ArtifactError as e:
        st.error(str(e))
        st.info("Go to **⚙️ Run Pipeline** in the sidebar to build the missing files from the raw dataset, "
                "or open **System check** to see which files are missing.")
    except Exception as e:  # unexpected problem: show it clearly instead of a blank page
        st.error(f"Something went wrong while building this page: {e}")
        with st.expander("Technical details"):
            st.exception(e)


main()
