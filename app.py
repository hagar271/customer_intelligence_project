"""Customer Intelligence System - Streamlit dashboard.

If generated artifacts are absent, every main dashboard page displays a clearly labelled
presentation preview instead of an artifact-missing error. No raw dataset upload is needed.
"""
import streamlit as st

st.set_page_config(page_title="Customer Intelligence System", page_icon="📊", layout="wide")

import data_loader  # noqa: E402
from utils import inject_css  # noqa: E402
import demo_dashboard  # noqa: E402
import page_churn as churn  # noqa: E402
import page_overview as overview  # noqa: E402
import page_prediction as prediction  # noqa: E402
import page_sales as sales  # noqa: E402
import page_segmentation as segmentation  # noqa: E402

PAGES = {
    "📊 Overview": overview.render,
    "🛒 Sales Analysis": sales.render,
    "🧩 Customer Segmentation": segmentation.render,
    "🔮 Churn Prediction": churn.render,
    "🎯 Customer Prediction": prediction.render,
}


def main():
    inject_css()
    missing = data_loader.missing_required()
    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Online Retail II")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()), index=0, key="nav",
                            label_visibility="collapsed")

    with st.sidebar.expander("System check (files & versions)"):
        st.dataframe(data_loader.artifact_status(), hide_index=True)
        if missing:
            st.caption("Generated model/result files are missing. Dashboard pages are currently in demo-preview mode.")
        elif data_loader.exists("churn_meta"):
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

    if missing:
        demo_dashboard.render(page)
        return

    try:
        PAGES[page]()
    except data_loader.ArtifactError as e:
        st.warning(f"Required generated project files are unavailable: {e}")
        st.info("Overview and the other pages can still be viewed in demo-preview mode if project artifacts are removed.")
    except Exception as e:
        st.error(f"Something went wrong while building this page: {e}")
        with st.expander("Technical details"):
            st.exception(e)


main()
