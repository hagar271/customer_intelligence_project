"""Customer Intelligence System - Streamlit dashboard.

When generated model/result artifacts are absent, the Overview page shows a clearly
labelled demo preview. The Pipeline & Deployment page is intentionally not in navigation.
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


def sidebar(missing):
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
        if missing:
            st.caption("Generated result/model files are missing. Overview will show a demo preview.")
    return page


def main():
    inject_css()
    missing = data_loader.missing_required()
    page = sidebar(missing)

    try:
        if missing and page == "📊 Overview":
            demo_dashboard.render()
        else:
            PAGES[page]()
    except data_loader.ArtifactError as e:
        st.warning(f"This page needs generated project result files that are not currently available. {e}")
        st.info("The Overview page still provides a dashboard preview without requiring the original dataset.")
    except Exception as e:
        st.error(f"Something went wrong while building this page: {e}")
        with st.expander("Technical details"):
            st.exception(e)


main()
