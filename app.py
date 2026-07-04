import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import networkx as nx

st.set_page_config(page_title="OlfactoSense Studio", page_icon="🧪", layout="wide")

st.markdown('''
<style>
.main {background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);}
.title {font-size: 3rem; font-weight: 900; line-height: 1.05; color: #0b1f3a; margin-bottom: .2rem;}
.subtitle {font-size: 1.15rem; color: #26415f; margin-bottom: 1.2rem;}
.metric-card {background: white; border: 1px solid #dbe7f3; border-radius: 18px; padding: 1rem; box-shadow: 0 4px 12px rgba(11,31,58,.06); height: 100%;}
.callout {background: #eef7ff; border-left: 6px solid #1f77b4; border-radius: 12px; padding: 1rem; color: #0b1f3a; margin: .5rem 0 1rem 0;}
.green-callout {background: #effaf2; border-left: 6px solid #2ca02c; border-radius: 12px; padding: 1rem; color: #0b1f3a; margin: .5rem 0 1rem 0;}
.orange-callout {background: #fff7ec; border-left: 6px solid #ff7f0e; border-radius: 12px; padding: 1rem; color: #0b1f3a; margin: .5rem 0 1rem 0;}
</style>
''', unsafe_allow_html=True)

np.random.seed(42)

VOCS = pd.DataFrame([
    ["Acetone", "Ketone", "C3H6O", 58.08, .90, .62, "carbonyl, methyl", "fatty acid oxidation, ketosis, breath metabolism", "metabolic stress, diabetes, fasting", .76, "CC(=O)C"],
    ["Ethanol", "Alcohol", "C2H6O", 46.07, .88, .70, "hydroxyl", "fermentation, microbiome, food processing", "fermentation, microbial metabolism", .67, "CCO"],
    ["Ammonia", "Amine/Nitrogen", "NH3", 17.03, .95, .85, "amine-like nitrogen", "protein degradation, renal metabolism, spoilage", "waste nitrogen, food spoilage, infection", .83, "N"],
    ["Hydrogen sulfide", "Sulfur", "H2S", 34.08, .96, .54, "reduced sulfur", "anaerobic microbes, rotten protein, sewage", "spoilage, anaerobic infection, environmental hazard", .92, "S"],
    ["Benzaldehyde", "Aldehyde", "C7H6O", 106.12, .54, .45, "aromatic aldehyde", "plant, flavour, microbial transformation", "food aroma, plant metabolism", .61, "O=CC1=CC=CC=C1"],
    ["2-Nonanone", "Ketone", "C9H18O", 142.24, .42, .34, "ketone, hydrocarbon chain", "microbial VOC, food headspace, C. elegans avoidance studies", "aversive odour, microbial context", .59, "CCCCCCCC(=O)C"],
    ["Dimethyl sulfide", "Sulfur", "C2H6S", 62.13, .86, .32, "thioether", "marine microbiome, spoilage, cabbage-like odour", "microbial metabolism, environment", .88, "CSC"],
    ["Limonene", "Terpene", "C10H16", 136.24, .46, .12, "terpene, alkene", "citrus, plants, cleaning products", "plant and indoor environment", .52, "CC1=CCC(CC1)C(=C)C"],
], columns=["VOC","Class","Formula","MW","Volatility","Polarity","Functional_groups","Origin","Biological_context","Sensor_affinity","SMILES_like"])

CONTEXTS = {
    "Safe background": {"ketone": .10, "aldehyde": .10, "alcohol": .16, "amine": .05, "sulfur": .03, "terpene": .20, "stress": .08, "hazard": .02},
    "Food spoilage": {"ketone": .35, "aldehyde": .28, "alcohol": .30, "amine": .62, "sulfur": .55, "terpene": .10, "stress": .55, "hazard": .35},
    "Infection-like VOC": {"ketone": .62, "aldehyde": .32, "alcohol": .22, "amine": .42, "sulfur": .25, "terpene": .05, "stress": .78, "hazard": .45},
    "Environmental hazard": {"ketone": .45, "aldehyde": .55, "alcohol": .38, "amine": .24, "sulfur": .78, "terpene": .12, "stress": .86, "hazard": .92},
    "Fermentation quality": {"ketone": .18, "aldehyde": .22, "alcohol": .72, "amine": .10, "sulfur": .08, "terpene": .28, "stress": .20, "hazard": .08},
}

SENSOR_WEIGHTS = pd.DataFrame({
    "MOS_1": {"ketone": .88, "aldehyde": .25, "alcohol": .26, "amine": .18, "sulfur": .12, "terpene": .16, "stress": .20, "hazard": .24},
    "MOS_2": {"ketone": .18, "aldehyde": .82, "alcohol": .18, "amine": .18, "sulfur": .32, "terpene": .18, "stress": .28, "hazard": .32},
    "MOS_3": {"ketone": .24, "aldehyde": .22, "alcohol": .84, "amine": .18, "sulfur": .20, "terpene": .28, "stress": .18, "hazard": .18},
    "EC_NH3": {"ketone": .05, "aldehyde": .12, "alcohol": .10, "amine": .88, "sulfur": .35, "terpene": .04, "stress": .42, "hazard": .40},
    "EC_H2S": {"ketone": .06, "aldehyde": .20, "alcohol": .10, "amine": .25, "sulfur": .94, "terpene": .04, "stress": .52, "hazard": .75},
    "QCM_Acid": {"ketone": .14, "aldehyde": .24, "alcohol": .24, "amine": .46, "sulfur": .26, "terpene": .10, "stress": .45, "hazard": .28},
    "PID": {"ketone": .52, "aldehyde": .50, "alcohol": .44, "amine": .30, "sulfur": .32, "terpene": .35, "stress": .25, "hazard": .62},
    "Optical": {"ketone": .20, "aldehyde": .32, "alcohol": .42, "amine": .16, "sulfur": .22, "terpene": .42, "stress": .18, "hazard": .22},
}).T

chem_cols = ["ketone","aldehyde","alcohol","amine","sulfur","terpene","stress","hazard"]

@st.cache_data
def simulate_dataset(n_per_context=180):
    rows = []
    for label, base in CONTEXTS.items():
        for i in range(n_per_context):
            conc = {k: max(0, np.random.normal(v, .06)) for k, v in base.items()}
            h = np.clip(np.random.normal(55, 9), 20, 90)
            t = np.clip(np.random.normal(22, 2.5), 10, 36)
            row = {"sample_id": f"{label.replace(' ', '_')}_{i:04d}", "context": label, "humidity": h, "temperature": t}
            row.update(conc)
            for sensor in SENSOR_WEIGHTS.index:
                signal = sum(SENSOR_WEIGHTS.loc[sensor, k] * conc[k] for k in SENSOR_WEIGHTS.columns)
                signal += .004*(h-55) + .010*(t-22) + np.random.normal(0, .07)
                row[f"sensor_{sensor}"] = max(0, signal)
            row["bio_attraction"] = np.clip(.82 - .65*conc["hazard"] - .40*conc["sulfur"] + np.random.normal(0,.05),0,1)
            row["bio_avoidance"] = np.clip(.15 + .75*conc["hazard"] + .30*conc["sulfur"] + np.random.normal(0,.05),0,1)
            row["bio_stress"] = np.clip(.10 + .80*conc["stress"] + .20*conc["amine"] + np.random.normal(0,.05),0,1)
            row["bio_meaning_score"] = np.clip(.35*row["bio_avoidance"] + .45*row["bio_stress"] + .20*conc["hazard"],0,1)
            rows.append(row)
    return pd.DataFrame(rows)

df = simulate_dataset()
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
bio_cols = ["bio_attraction","bio_avoidance","bio_stress","bio_meaning_score"]

def train_model(_data):
    feature_cols = sensor_cols + chem_cols + ["humidity", "temperature"] + bio_cols
    X, y = _data[feature_cols], _data["context"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.25)
    model = RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test,pred), "balanced_accuracy": balanced_accuracy_score(y_test,pred), "macro_f1": f1_score(y_test,pred,average="macro")}
    labels = sorted(_data["context"].unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    imp = permutation_importance(model, X_test, y_test, random_state=42, n_repeats=5, scoring="balanced_accuracy")
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    return model, feature_cols, metrics, cm, labels, imp_df

model, feature_cols, metrics, cm, class_labels, imp_df = train_model(df)


def build_current_sample(selected_domain, humidity_value, temperature_value, drift_value, noise_value):
    """Build one simulated olfactory sample for prediction and sensor visualisation."""
    base_profile = CONTEXTS[selected_domain].copy()

    sim_row_local = {}
    for key, value in base_profile.items():
        sim_row_local[key] = max(0, np.random.normal(value, 0.02))

    sensor_response_local = {}
    for sensor in SENSOR_WEIGHTS.index:
        signal = sum(
            SENSOR_WEIGHTS.loc[sensor, key] * sim_row_local[key]
            for key in SENSOR_WEIGHTS.columns
        )
        signal += 0.004 * (humidity_value - 55)
        signal += 0.010 * (temperature_value - 22)
        signal += drift_value
        signal += np.random.normal(0, noise_value)
        sensor_response_local[sensor] = max(0, signal)

    current_local = pd.DataFrame([{
        **sim_row_local,
        "humidity": humidity_value,
        "temperature": temperature_value,
        **{f"sensor_{key}": value for key, value in sensor_response_local.items()},
    }])

    current_local["bio_attraction"] = np.clip(
        0.82 - 0.65 * current_local["hazard"] - 0.40 * current_local["sulfur"],
        0,
        1,
    )
    current_local["bio_avoidance"] = np.clip(
        0.15 + 0.75 * current_local["hazard"] + 0.30 * current_local["sulfur"],
        0,
        1,
    )
    current_local["bio_stress"] = np.clip(
        0.10 + 0.80 * current_local["stress"] + 0.20 * current_local["amine"],
        0,
        1,
    )
    current_local["bio_meaning_score"] = np.clip(
        0.35 * current_local["bio_avoidance"]
        + 0.45 * current_local["bio_stress"]
        + 0.20 * current_local["hazard"],
        0,
        1,
    )

    return sim_row_local, sensor_response_local, current_local


st.sidebar.title("OlfactoSense Studio")
st.sidebar.caption("Molecular-to-systems digital olfaction")
domain = st.sidebar.selectbox("Scenario", list(CONTEXTS.keys()), index=2)
humidity = st.sidebar.slider("Humidity (%)", 20, 90, 55)
temperature = st.sidebar.slider("Temperature (°C)", 10, 36, 22)
sensor_drift = st.sidebar.slider("Sensor drift", 0.00, 0.25, 0.05, 0.01)
noise_level = st.sidebar.slider("Noise level", 0.00, 0.25, 0.06, 0.01)

st.markdown('<div class="title">OlfactoSense Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Molecular-to-Systems AI Platform for Digital Olfaction</div>', unsafe_allow_html=True)

PAGE_OPTIONS = [
    "Programme Thesis",
    "Molecular VOC Atlas",
    "Mixture Representation Map",
    "Sensor Array Simulator",
    "Biological Meaning Model",
    "AI Prediction and Uncertainty",
    "OlfactoBot Plume Mapping",
    "Health, Food, Environment",
    "Creator Proposal Evaluator",
    "GC-MS Raw Signal Viewer",
    "VOC Database Connector",
    "GNN Molecular Encoder",
    "Mixture Decoder",
    "Sensor Drift Laboratory",
    "Open Benchmark Builder",
    "Dataset Card Generator",
    "Model Card Generator",
    "Hardware Readiness Assessment",
]

page = st.sidebar.radio(
    "App module",
    PAGE_OPTIONS,
    index=1,
)


if page == PAGE_OPTIONS[0]:
    st.header("Programme Thesis")
    st.markdown('<div class="callout"><b>A molecule is not a smell by itself.</b> A smell is a dynamic, context-dependent chemical signal. To make it useful for AI, we need to capture the molecule, the mixture, the sensor response, the environment, the biological meaning, and the decision.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,title,val in [(c1,"Molecular level","VOC structure"),(c2,"Sensor level","Cross-reactive arrays"),(c3,"AI level","Representation learning"),(c4,"Systems level","Action and deployment")]:
        col.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{val}</p></div>', unsafe_allow_html=True)
    stages = ["VOC molecule","Mixture","Sensor response","AI representation","Biological meaning","Decision","Robot action"]
    fig = go.Figure()
    for i,s in enumerate(stages):
        fig.add_trace(go.Scatter(x=[i], y=[0], mode="markers+text", marker=dict(size=42), text=[s], textposition="bottom center", name=s))
        if i < len(stages)-1:
            fig.add_annotation(x=i+.5,y=0,ax=i+.1,ay=0,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=3)
    fig.update_layout(height=300, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

if page == PAGE_OPTIONS[1]:
    st.header("Molecular VOC Atlas")
    left,right = st.columns([1.05,1.45])
    with left:
        selected = st.selectbox("Select VOC", VOCS["VOC"])
        voc = VOCS[VOCS["VOC"]==selected].iloc[0]
        st.markdown(f"### {voc['VOC']}")
        st.table(voc[["Class","Formula","MW","Volatility","Polarity","Functional_groups","Origin","Biological_context","SMILES_like"]].to_frame("Value").astype(str))
    with right:
        fig = px.scatter(VOCS, x="Volatility", y="Polarity", size="Sensor_affinity", color="Class", hover_data=["VOC","Origin","Biological_context"], title="VOC molecular property space")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    G = nx.Graph()
    for _,r in VOCS.iterrows():
        G.add_node(r["VOC"], group="VOC"); G.add_node(r["Class"], group="Class"); G.add_edge(r["VOC"], r["Class"])
        for token in str(r["Biological_context"]).split(","):
            token=token.strip(); G.add_node(token, group="Meaning"); G.add_edge(r["VOC"], token)
    pos=nx.spring_layout(G, seed=4, k=.65)
    edge_x=[]; edge_y=[]
    for a,b in G.edges(): edge_x += [pos[a][0],pos[b][0],None]; edge_y += [pos[a][1],pos[b][1],None]
    colors={"VOC":"#1f77b4","Class":"#ff7f0e","Meaning":"#2ca02c"}
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=1,color="#bbb"),hoverinfo="none"))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], mode="markers+text", text=list(G.nodes()), textposition="top center", marker=dict(size=16, color=[colors.get(G.nodes[n]["group"],"#888") for n in G.nodes()])))
    fig.update_layout(height=520, showlegend=False, title="VOC, chemical class, and biological-origin network", xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

if page == PAGE_OPTIONS[2]:
    st.header("Mixture Representation Map")
    st.markdown('<div class="green-callout">This map simulates the representation-learning problem: a space where related chemical contexts cluster together while the same representation can transfer across health, food, and environmental domains.</div>', unsafe_allow_html=True)
    Xs = StandardScaler().fit_transform(df[chem_cols + sensor_cols + bio_cols])
    pcs = PCA(n_components=3, random_state=42).fit_transform(Xs)
    pca_df = df[["sample_id","context"]].copy(); pca_df["PC1"]=pcs[:,0]; pca_df["PC2"]=pcs[:,1]; pca_df["PC3"]=pcs[:,2]
    c1,c2 = st.columns([1.2,1])
    with c1:
        fig=px.scatter(pca_df, x="PC1", y="PC2", color="context", hover_name="sample_id", title="2D olfactory representation map")
        fig.update_layout(height=560); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig=px.scatter_3d(pca_df.sample(700, random_state=42), x="PC1", y="PC2", z="PC3", color="context", title="3D representation view")
        fig.update_layout(height=560); st.plotly_chart(fig, use_container_width=True)

if page == PAGE_OPTIONS[3]:
    st.header("Sensor Array Simulator")
    base = CONTEXTS[domain].copy()
    sim_row = {k:max(0,np.random.normal(v,.02)) for k,v in base.items()}
    sensor_response={}
    for sensor in SENSOR_WEIGHTS.index:
        val=sum(SENSOR_WEIGHTS.loc[sensor,k]*sim_row[k] for k in SENSOR_WEIGHTS.columns)
        val += .004*(humidity-55)+.010*(temperature-22)+sensor_drift+np.random.normal(0,noise_level)
        sensor_response[sensor]=max(0,val)
    sensor_df=pd.DataFrame({"Sensor":list(sensor_response.keys()),"Response":list(sensor_response.values())})
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(sensor_df,x="Sensor",y="Response",title=f"Cross-reactive sensor response: {domain}"); fig.update_layout(height=480); st.plotly_chart(fig,use_container_width=True)
    with c2:
        time=np.linspace(0,60,160); traces=[]
        for sensor,amp in sensor_response.items():
            rise=1-np.exp(-time/8); decay=np.exp(-np.maximum(time-35,0)/18)
            traces.append(pd.DataFrame({"time":time,"response":amp*rise*decay+np.random.normal(0,.01,len(time)),"sensor":sensor}))
        fig=px.line(pd.concat(traces),x="time",y="response",color="sensor",title="Dynamic sensor traces, rise and recovery"); fig.update_layout(height=480); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(sensor_df, use_container_width=True)

if page == PAGE_OPTIONS[4]:
    st.header("Biological Meaning Model")
    st.markdown('<div class="callout">Biological olfaction teaches that smell is chemical meaning linked to action. Inspired by simple systems such as <i>C. elegans</i>, this layer models attraction, avoidance, stress, and metabolic shift as functional labels for volatile chemistry.</div>', unsafe_allow_html=True)
    bio_summary=df.groupby("context")[bio_cols].mean().reset_index()
    fig=px.bar(bio_summary,x="context",y=bio_cols,barmode="group",title="Biological meaning scores by context"); fig.update_layout(height=560); st.plotly_chart(fig,use_container_width=True)
    st.write("AWA/AWC-like attraction signals represent food-associated contexts. AWB/ASH-like avoidance signals represent harmful or stress-associated contexts. These biological labels complement human odour descriptors.")

if page == PAGE_OPTIONS[5]:
    st.header("AI Prediction and Uncertainty")
    sim_row, sensor_response, current = build_current_sample(domain, humidity, temperature, sensor_drift, noise_level)
    current["bio_attraction"]=np.clip(.82-.65*current["hazard"]-.40*current["sulfur"],0,1)
    current["bio_avoidance"]=np.clip(.15+.75*current["hazard"]+.30*current["sulfur"],0,1)
    current["bio_stress"]=np.clip(.10+.80*current["stress"]+.20*current["amine"],0,1)
    current["bio_meaning_score"]=np.clip(.35*current["bio_avoidance"]+.45*current["bio_stress"]+.20*current["hazard"],0,1)
    proba=model.predict_proba(current[feature_cols])[0]; pred_label=model.classes_[np.argmax(proba)]; confidence=float(np.max(proba)); uncertainty=float(np.clip(1-confidence+.35*sensor_drift+.20*noise_level,0,1))
    c1,c2,c3=st.columns(3); c1.metric("Predicted meaning",pred_label); c2.metric("Confidence",f"{confidence:.2f}"); c3.metric("Uncertainty",f"{uncertainty:.2f}")
    fig=px.bar(pd.DataFrame({"Context":model.classes_,"Probability":proba}).sort_values("Probability",ascending=False),x="Context",y="Probability",title="AI prediction probabilities"); fig.update_layout(height=420); st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.imshow(pd.DataFrame(cm,index=class_labels,columns=class_labels),text_auto=True,title="Model validation confusion matrix"); fig.update_layout(height=520); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.bar(imp_df.head(18).iloc[::-1],x="importance",y="feature",orientation="h",title="Feature importance"); fig.update_layout(height=520); st.plotly_chart(fig,use_container_width=True)
    st.json({k:round(v,3) for k,v in metrics.items()})

if page == PAGE_OPTIONS[6]:
    st.header("OlfactoBot Plume Mapping")
    width,height=40,30; source_x,source_y=31,21; steps=st.slider("Mission steps",20,120,70)
    path=[]; x,y=3.0,3.0
    for step in range(steps):
        d=math.sqrt((x-source_x)**2+(y-source_y)**2); strength=np.clip(1.4*math.exp(-d/8.0)*(1+.02*(source_x-x)),.02,1.5)
        if strength < .06:
            dx=np.sign(source_x-x)*1.05+np.random.normal(0,.2); dy=np.sign(source_y-y)*.85+np.random.normal(0,.2); action="search and reacquire"
        elif strength > .18:
            vx,vy=x-source_x,y-source_y; norm=max(math.sqrt(vx*vx+vy*vy),.1); radial=np.array([vx/norm,vy/norm]); tangent=np.array([-radial[1],radial[0]]); dx,dy=1.1*radial+.8*tangent; action="safe standoff mapping"
        else:
            vx,vy=x-source_x,y-source_y; norm=max(math.sqrt(vx*vx+vy*vy),.1); radial=np.array([vx/norm,vy/norm]); tangent=np.array([-radial[1],radial[0]]); dx,dy=1.2*tangent; action="orbit plume boundary"
        path.append({"step":step,"x":x,"y":y,"plume_strength":strength,"action":action})
        x=float(np.clip(x+dx,2,width-2)); y=float(np.clip(y+dy,2,height-2))
    path_df=pd.DataFrame(path)
    c1,c2=st.columns([1.1,1])
    with c1:
        fig=px.scatter(path_df,x="x",y="y",color="plume_strength",symbol="action",title="Robot plume-boundary mapping path")
        fig.add_trace(go.Scatter(x=[source_x],y=[source_y],mode="markers+text",marker=dict(size=24,symbol="star"),text=["source"],textposition="top center",name="chemical source"))
        fig.update_layout(height=600,xaxis_range=[0,width],yaxis_range=[0,height]); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.line(path_df,x="step",y="plume_strength",color="action",markers=True,title="Plume exposure timeline"); fig.update_layout(height=600); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(path_df.tail(15), use_container_width=True)

if page == PAGE_OPTIONS[7]:
    st.header("Health, Food, Environment Use Cases")
    use_cases=pd.DataFrame([
        ["Health","Breath VOC monitoring","Longitudinal disease flare risk, metabolic stress, infection-like VOCs","Non-invasive screening and follow-up"],
        ["Health","Skin or sebum VOCs","Inflammation, Parkinson-like sebum chemistry, oxidative stress","Early warning and stratification"],
        ["Food","Freshness and spoilage","Amines, sulfur volatiles, aldehydes, acids","Replace blunt expiry dates with real-time quality signals"],
        ["Food","Fermentation monitoring","Alcohols, acids, esters, microbial VOCs","Quality control and process optimisation"],
        ["Environment","Indoor exposome","Mould, solvents, combustion, cleaning product VOCs","Air-quality and exposure mapping"],
        ["Environment","Hazard detection","Sulfur compounds, solvents, industrial emissions","Safety alerts and plume localisation"],
    ],columns=["Domain","Use case","Chemical signal","Decision value"])
    fig=px.sunburst(use_cases,path=["Domain","Use case"],values=[1]*len(use_cases),title="Cross-domain artificial olfaction use cases"); fig.update_layout(height=650); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(use_cases,use_container_width=True)

if page == PAGE_OPTIONS[8]:
    st.header("Creator Proposal Evaluator")
    st.markdown('<div class="orange-callout">Score a hypothetical Creator proposal against the factors needed for a general-purpose olfactory perception system rather than a narrow point solution.</div>', unsafe_allow_html=True)
    criteria=["Dataset quality","Sensor robustness","Calibration strategy","Representation learning value","Cross-domain transfer","Biological relevance","Deployability","Uncertainty handling","Milestone credibility"]
    scores={}; cols=st.columns(3)
    for i,crit in enumerate(criteria):
        with cols[i%3]: scores[crit]=st.slider(crit,0,10,7)
    score_df=pd.DataFrame({"Criterion":list(scores.keys()),"Score":list(scores.values())}); total=score_df["Score"].mean()
    c1,c2=st.columns(2)
    with c1:
        st.metric("Portfolio fit score",f"{total:.1f}/10")
        if total >= 8: st.success("Strong general-purpose olfaction fit. Worth serious technical diligence.")
        elif total >= 6: st.warning("Promising but needs tighter milestones, validation, or cross-domain evidence.")
        else: st.error("Likely a narrow point solution or insufficiently validated proposal.")
    with c2:
        fig=go.Figure(); fig.add_trace(go.Scatterpolar(r=score_df["Score"],theta=score_df["Criterion"],fill="toself",name="Proposal")); fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,10])),showlegend=False,title="Creator proposal radar"); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Suggested reviewer questions")
    st.write("""
    - Does the dataset include enough metadata to support transfer learning and external validation?
    - Are sensor drift, humidity, temperature, and sample preparation explicitly controlled?
    - Does the representation learn reusable olfactory structure or only classify one dataset?
    - Can the hardware resolve the discriminative dimensions identified by the model?
    - Are biological or functional labels available beyond subjective odour descriptors?
    - What milestone would falsify the central technical claim?
    """)



# -----------------------------
# 10 GC-MS Raw Signal Viewer
# -----------------------------
if page == PAGE_OPTIONS[9]:
    st.header("GC-MS Raw Signal Viewer")
    st.markdown("""
    <div class="callout">
    Simulated GC-MS chromatogram and mass-spectrum viewer. This represents the high-resolution
    analytical chemistry layer that can anchor sensor-array data and olfactory representation learning.
    </div>
    """, unsafe_allow_html=True)

    gc_context = st.selectbox("Select analytical sample context", list(CONTEXTS.keys()), index=2, key="gc_context_select")

    rt_axis = np.linspace(0, 20, 1200)
    voc_peaks = {
        "Acetone": {"rt": 2.1, "mz": [43, 58], "weights": [1.0, 0.55]},
        "Ethanol": {"rt": 1.4, "mz": [31, 45, 46], "weights": [1.0, 0.44, 0.30]},
        "Ammonia": {"rt": 0.8, "mz": [17], "weights": [1.0]},
        "Hydrogen sulfide": {"rt": 1.1, "mz": [34], "weights": [1.0]},
        "Benzaldehyde": {"rt": 8.9, "mz": [77, 105, 106], "weights": [0.62, 1.0, 0.45]},
        "2-Nonanone": {"rt": 12.4, "mz": [43, 58, 142], "weights": [1.0, 0.72, 0.24]},
        "Dimethyl sulfide": {"rt": 3.3, "mz": [47, 62], "weights": [0.65, 1.0]},
        "Limonene": {"rt": 14.8, "mz": [68, 93, 136], "weights": [0.74, 1.0, 0.30]},
    }

    context_profile = CONTEXTS[gc_context]
    class_to_intensity = {
        "Ketone": context_profile["ketone"],
        "Alcohol": context_profile["alcohol"],
        "Amine/Nitrogen": context_profile["amine"],
        "Sulfur": context_profile["sulfur"],
        "Aldehyde": context_profile["aldehyde"],
        "Terpene": context_profile["terpene"],
    }

    chromatogram = np.zeros_like(rt_axis)
    ms_rows = []
    peak_rows = []

    for _, r in VOCS.iterrows():
        peak = voc_peaks[r["VOC"]]
        amp = float(class_to_intensity.get(r["Class"], 0.2)) * float(r["Sensor_affinity"])
        width = 0.08 + 0.02 * np.random.rand()
        chromatogram += amp * np.exp(-0.5 * ((rt_axis - peak["rt"]) / width) ** 2)

        peak_rows.append({
            "VOC": r["VOC"],
            "Retention time, min": peak["rt"],
            "Class": r["Class"],
            "Relative abundance": amp,
            "Likely origin": r["Origin"],
        })

        for mz, w in zip(peak["mz"], peak["weights"]):
            ms_rows.append({
                "VOC": r["VOC"],
                "m/z": mz,
                "Intensity": amp * w,
                "Class": r["Class"],
            })

    chromatogram += np.random.normal(0, 0.006, len(chromatogram))

    chrom_df = pd.DataFrame({
        "Retention time, min": rt_axis,
        "Total ion current": chromatogram,
    })

    ms_df = pd.DataFrame(ms_rows)
    peak_df = pd.DataFrame(peak_rows).sort_values("Relative abundance", ascending=False)

    c1, c2 = st.columns([1.3, 1])

    with c1:
        fig = px.line(
            chrom_df,
            x="Retention time, min",
            y="Total ion current",
            title=f"Simulated GC-MS chromatogram: {gc_context}",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        selected_peak = st.selectbox("Inspect mass spectrum for VOC", peak_df["VOC"], key="gc_peak_select")
        spec = ms_df[ms_df["VOC"] == selected_peak]
        fig = px.bar(spec, x="m/z", y="Intensity", title=f"Mass spectrum fragments: {selected_peak}")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected peak table")
    st.dataframe(peak_df.astype(str), use_container_width=True)


# -----------------------------
# 11 VOC Database Connector
# -----------------------------
if page == PAGE_OPTIONS[10]:
    st.header("VOC Database Connector")
    st.markdown("""
    <div class="green-callout">
    Prototype connector for HMDB, mVOC, FooDB, Cancer Odor Database, and OlfactionBase.
    This portable version uses embedded simulated evidence tables rather than live API calls.
    </div>
    """, unsafe_allow_html=True)

    sources = ["HMDB", "mVOC", "FooDB", "Cancer Odor Database", "OlfactionBase"]
    rows = []

    for _, r in VOCS.iterrows():
        for src in sources:
            relevance = np.clip(
                0.25
                + 0.30 * (src == "mVOC" and "micro" in r["Origin"].lower())
                + 0.30 * (src == "FooDB" and ("food" in r["Origin"].lower() or "flavour" in r["Origin"].lower()))
                + 0.30 * (src == "HMDB" and ("breath" in r["Origin"].lower() or "metabolism" in r["Origin"].lower()))
                + 0.30 * (src == "Cancer Odor Database" and ("stress" in r["Biological_context"].lower() or "metabolic" in r["Biological_context"].lower()))
                + 0.20 * np.random.rand(),
                0,
                1,
            )
            rows.append({
                "VOC": r["VOC"],
                "Database": src,
                "Predicted relevance": relevance,
                "Evidence type": np.random.choice(["literature association", "metabolite record", "food VOC", "microbial VOC", "olfactory descriptor"]),
                "Application link": r["Biological_context"],
            })

    db_df = pd.DataFrame(rows)

    query = st.text_input("Search VOC, database, origin, or application", value="")
    filtered = db_df.copy()
    if query.strip():
        q = query.lower()
        filtered = filtered[filtered.apply(lambda row: q in " ".join(map(str, row.values)).lower(), axis=1)]

    c1, c2 = st.columns([1.2, 1])

    with c1:
        fig = px.bar(
            db_df.groupby("Database")["Predicted relevance"].mean().reset_index(),
            x="Database",
            y="Predicted relevance",
            title="Mean simulated database relevance",
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        heat = db_df.pivot_table(index="VOC", columns="Database", values="Predicted relevance", aggfunc="mean")
        fig = px.imshow(heat, text_auto=".2f", title="VOC-database evidence map")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered.sort_values("Predicted relevance", ascending=False), use_container_width=True)


# -----------------------------
# 12 Graph Neural Network Molecular Encoder
# -----------------------------
if page == PAGE_OPTIONS[11]:
    st.header("Graph Neural Network Molecular Encoder")
    st.markdown("""
    <div class="callout">
    Lightweight molecular-graph demonstrator. It approximates how a graph neural network could encode
    atoms, functional groups, and molecular properties into an olfactory embedding.
    </div>
    """, unsafe_allow_html=True)

    selected_mol = st.selectbox("Select molecule for graph encoding", VOCS["VOC"], key="gnn_mol_select")
    mol = VOCS[VOCS["VOC"] == selected_mol].iloc[0]

    smiles = mol["SMILES_like"]
    atoms = [ch for ch in smiles if ch.isalpha() and ch.isupper()]
    if not atoms:
        atoms = ["C", "O"]

    G = nx.path_graph(len(atoms))
    labels = {i: atoms[i] for i in range(len(atoms))}
    pos = nx.spring_layout(G, seed=7)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        node_text.append(labels[n])

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=3), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="middle center", marker=dict(size=42)))
        fig.update_layout(title=f"Approximate molecular graph: {selected_mol}", height=520, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

    all_graph = VOCS.copy()
    all_graph["n_atoms_proxy"] = all_graph["SMILES_like"].apply(lambda s: sum(1 for ch in s if ch.isalpha() and ch.isupper()))
    all_graph["heteroatom_proxy"] = all_graph["SMILES_like"].apply(lambda s: sum(1 for ch in s if ch in ["O", "N", "S"]))
    all_graph["carbon_proxy"] = all_graph["SMILES_like"].apply(lambda s: sum(1 for ch in s if ch == "C"))

    emb_cols = ["MW", "Volatility", "Polarity", "Sensor_affinity", "n_atoms_proxy", "heteroatom_proxy", "carbon_proxy"]
    emb = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(all_graph[emb_cols]))

    emb_df = all_graph[["VOC", "Class", "Biological_context"]].copy()
    emb_df["GNN dimension 1"] = emb[:, 0]
    emb_df["GNN dimension 2"] = emb[:, 1]

    with c2:
        fig = px.scatter(
            emb_df,
            x="GNN dimension 1",
            y="GNN dimension 2",
            color="Class",
            hover_data=["VOC", "Biological_context"],
            title="Molecular embedding proxy",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 13 Mixture Decoder
# -----------------------------
if page == PAGE_OPTIONS[12]:
    st.header("Mixture Decoder")
    st.markdown("""
    <div class="green-callout">
    Predict mixture similarity and functional olfactory meaning by comparing chemical composition,
    sensor response, and biological labels.
    </div>
    """, unsafe_allow_html=True)

    mix_a = st.selectbox("Mixture A", list(CONTEXTS.keys()), index=0, key="mix_a")
    mix_b = st.selectbox("Mixture B", list(CONTEXTS.keys()), index=2, key="mix_b")

    vec_a = np.array([CONTEXTS[mix_a][k] for k in chem_cols])
    vec_b = np.array([CONTEXTS[mix_b][k] for k in chem_cols])

    cosine = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
    euclid = float(np.linalg.norm(vec_a - vec_b))
    transfer_score = float(np.clip(1 - euclid / 2.2, 0, 1))

    comparison = pd.DataFrame({
        "Chemical feature": chem_cols,
        mix_a: vec_a,
        mix_b: vec_b,
        "Absolute difference": np.abs(vec_a - vec_b),
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Mixture similarity", f"{cosine:.2f}")
    c2.metric("Chemical distance", f"{euclid:.2f}")
    c3.metric("Cross-domain transfer potential", f"{transfer_score:.2f}")

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = px.bar(comparison, x="Chemical feature", y=[mix_a, mix_b], barmode="group", title="Mixture composition comparison")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(comparison.sort_values("Absolute difference"), x="Absolute difference", y="Chemical feature", orientation="h", title="Discriminative mixture dimensions")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    if transfer_score > 0.75:
        st.success("High similarity. A model trained on one mixture may transfer well to the other.")
    elif transfer_score > 0.45:
        st.warning("Partial overlap. Transfer may require calibration, domain adaptation, or additional labels.")
    else:
        st.error("Low similarity. Useful for testing whether a representation truly generalises.")


# -----------------------------
# 14 Sensor Drift Laboratory
# -----------------------------
if page == PAGE_OPTIONS[13]:
    st.header("Sensor Drift Laboratory")
    st.markdown("""
    <div class="callout">
    Sensor drift is a central barrier in deployable digital olfaction. This module simulates baseline drift,
    reference correction, and calibration stability gain.
    </div>
    """, unsafe_allow_html=True)

    days = st.slider("Simulated deployment days", 10, 120, 60, key="drift_days")
    drift_strength = st.slider("Drift strength", 0.00, 0.80, 0.35, 0.01, key="drift_lab_strength")

    day_axis = np.arange(days)
    drift_rows = []

    for sensor in SENSOR_WEIGHTS.index:
        baseline = np.random.uniform(0.2, 0.8)
        raw = baseline + drift_strength * (day_axis / max(days, 1)) + 0.08 * np.sin(day_axis / 8) + np.random.normal(0, 0.03, days)
        corrected = raw - pd.Series(raw).rolling(7, min_periods=1).mean() + baseline

        for d, rraw, cor in zip(day_axis, raw, corrected):
            drift_rows.append({"Day": d, "Sensor": sensor, "Raw signal": rraw, "Corrected signal": cor})

    drift_df = pd.DataFrame(drift_rows)
    selected_sensor = st.selectbox("Sensor", SENSOR_WEIGHTS.index, key="drift_sensor_select")
    sdf = drift_df[drift_df["Sensor"] == selected_sensor]

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = px.line(sdf, x="Day", y=["Raw signal", "Corrected signal"], title=f"Drift correction: {selected_sensor}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        summary = drift_df.groupby("Sensor").agg(
            raw_sd=("Raw signal", "std"),
            corrected_sd=("Corrected signal", "std"),
        ).reset_index()
        summary["Stability gain"] = 1 - summary["corrected_sd"] / summary["raw_sd"]

        fig = px.bar(summary, x="Sensor", y="Stability gain", title="Estimated calibration stability gain")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True)


# -----------------------------
# 15 Open Benchmark Builder
# -----------------------------
if page == PAGE_OPTIONS[14]:
    st.header("Open Benchmark Builder")
    st.markdown("""
    <div class="green-callout">
    Design olfaction benchmark tasks that test generality, transfer, robustness, calibration,
    and real-world decision value.
    </div>
    """, unsafe_allow_html=True)

    benchmark_name = st.text_input("Benchmark name", "Cross-domain VOC representation challenge")

    target_domains = st.multiselect(
        "Target domains",
        ["longitudinal health", "food spoilage", "food safety", "indoor environment", "microbial VOCs", "industrial hazard"],
        default=["longitudinal health", "food spoilage", "indoor environment"],
    )

    instruments = st.multiselect(
        "Data sources",
        ["GC-MS", "PTR-MS", "SIFT-MS", "MOS array", "QCM array", "electrochemical sensors", "optical array", "biohybrid receptor"],
        default=["GC-MS", "MOS array", "electrochemical sensors"],
    )

    metrics_selected = st.multiselect(
        "Benchmark metrics",
        ["balanced accuracy", "AUROC", "calibration error", "out-of-distribution detection", "cross-site transfer", "sensor drift robustness", "sample efficiency", "interpretability"],
        default=["balanced accuracy", "calibration error", "cross-site transfer", "sensor drift robustness"],
    )

    n_samples = st.slider("Target number of samples", 1000, 1000000, 50000, step=1000)

    difficulty = min(10, 2 + len(target_domains) + len(instruments) * 0.5 + len(metrics_selected) * 0.4 + np.log10(n_samples) / 2)
    utility = min(10, 3 + len(target_domains) * 0.9 + len(metrics_selected) * 0.5 + ("GC-MS" in instruments) * 1.0)

    c1, c2 = st.columns(2)
    c1.metric("Benchmark difficulty", f"{difficulty:.1f}/10")
    c2.metric("Expected field utility", f"{utility:.1f}/10")

    benchmark_card = pd.DataFrame({
        "Field": ["Benchmark", "Domains", "Instruments", "Metrics", "Target samples", "Main scientific question"],
        "Value": [
            benchmark_name,
            ", ".join(target_domains),
            ", ".join(instruments),
            ", ".join(metrics_selected),
            f"{n_samples:,}",
            "Can a learned olfactory representation transfer across chemical contexts without rebuilding hardware?",
        ],
    })

    st.dataframe(benchmark_card.astype(str), use_container_width=True)


# -----------------------------
# 16 Dataset Card Generator
# -----------------------------
if page == PAGE_OPTIONS[15]:
    st.header("Dataset Card Generator")
    st.markdown("""
    <div class="callout">
    Generate metadata and governance documentation for olfactory datasets.
    </div>
    """, unsafe_allow_html=True)

    ds_name = st.text_input("Dataset name", "OlfactoSense Cross-Domain VOC Dataset")
    ds_owner = st.text_input("Dataset owner", "aAidea Ltd / research demonstrator")

    ds_domains = st.multiselect(
        "Dataset domains",
        ["health", "food spoilage", "food safety", "environment", "microbial VOCs", "robot plume mapping"],
        default=["health", "food spoilage", "environment"],
    )

    ds_instruments = st.multiselect(
        "Instrument types",
        ["GC-MS", "PTR-MS", "SIFT-MS", "MOS array", "QCM array", "electrochemical array", "optical array"],
        default=["GC-MS", "MOS array", "electrochemical array"],
        key="ds_instruments",
    )

    ds_limitations = st.text_area(
        "Known limitations",
        "Prototype uses simulated data. Real deployment would require analytical validation, calibration transfer, batch metadata, independent cohorts, and regulatory review."
    )

    dataset_card = f"""# Dataset Card: {ds_name}

## Dataset owner
{ds_owner}

## Dataset purpose
This dataset is intended to support artificial olfaction research, representation learning, sensor benchmarking, and cross-domain VOC analysis.

## Domains
{", ".join(ds_domains)}

## Instruments
{", ".join(ds_instruments)}

## Data modalities
- VOC molecular or chemical features
- Sensor-array responses
- Environmental metadata
- Biological or functional labels
- Application-relevant outcomes

## Metadata requirements
- Sample source
- Collection protocol
- Instrument type
- Batch and site
- Temperature
- Humidity
- Storage and transport conditions
- Calibration controls
- Label provenance

## Intended use
Research, benchmarking, model development, representation learning, and technical evaluation.

## Out-of-scope use
This dataset should not be used for clinical diagnosis, food safety certification, environmental safety certification, or regulatory claims without independent validation.

## Known limitations
{ds_limitations}

## Governance notes
Future real datasets should include consent, privacy review, sample governance, access controls, and documentation of potential misuse.
"""

    st.download_button("Download dataset card Markdown", dataset_card, file_name="DATASET_CARD.md", mime="text/markdown")
    st.code(dataset_card, language="markdown")


# -----------------------------
# 17 Model Card Generator
# -----------------------------
if page == PAGE_OPTIONS[16]:
    st.header("Model Card Generator")
    st.markdown("""
    <div class="green-callout">
    Generate model documentation for intended use, model inputs, uncertainty, explainability,
    limitations, and failure modes.
    </div>
    """, unsafe_allow_html=True)

    model_name = st.text_input("Model name", "OlfactoSense Random Forest Context Classifier")

    intended_use = st.text_area(
        "Intended use",
        "Research demonstration of functional VOC context prediction from simulated sensor, chemistry, environmental, and biological meaning features."
    )

    failure_modes = st.text_area(
        "Key failure modes",
        "Sensor drift, humidity confounding, batch effects, weak labels, unseen mixtures, poor calibration, and extrapolation beyond simulated domains."
    )

    model_card = f"""# Model Card: {model_name}

## Model type
Random Forest classifier prototype.

## Intended use
{intended_use}

## Inputs
- Simulated sensor-array responses
- Chemical feature intensities
- Humidity and temperature
- Biological meaning scores

## Outputs
- Predicted olfactory context
- Class probabilities
- Confidence estimate
- Uncertainty score

## Current validation metrics
- Accuracy: {metrics["accuracy"]:.3f}
- Balanced accuracy: {metrics["balanced_accuracy"]:.3f}
- Macro F1: {metrics["macro_f1"]:.3f}

## Explainability
Feature importance is estimated using permutation importance. The app also displays class probabilities and confusion matrix.

## Uncertainty
Uncertainty is approximated from prediction confidence, noise, and drift parameters. This is a prototype score, not a calibrated probabilistic guarantee.

## Key failure modes
{failure_modes}

## Ethical and safety notes
This model is not a diagnostic, food safety, or environmental safety tool. It is for research design, portfolio demonstration, and technical exploration.

## Recommended validation before real use
- External validation across sites
- Analytical chemistry confirmation
- Sensor calibration transfer testing
- Out-of-distribution detection
- Prospective evaluation
- Independent audit of labels and metadata
"""

    st.download_button("Download model card Markdown", model_card, file_name="MODEL_CARD.md", mime="text/markdown")
    st.code(model_card, language="markdown")


# -----------------------------
# 18 Hardware Readiness Assessment
# -----------------------------
if page == PAGE_OPTIONS[17]:
    st.header("Hardware Readiness Assessment")
    st.markdown("""
    <div class="orange-callout">
    Compare olfactory sensing platforms across sensitivity, selectivity, portability, cost,
    throughput, drift resistance, and chemical insight.
    </div>
    """, unsafe_allow_html=True)

    platforms = ["MOS array", "Conducting polymer", "QCM", "Electrochemical", "Optical array", "Ion mobility", "Micro-GC", "GC-MS", "Biohybrid receptor"]

    default_scores = {
        "MOS array": [6, 4, 9, 9, 9, 3, 2],
        "Conducting polymer": [5, 4, 9, 8, 9, 3, 2],
        "QCM": [7, 5, 8, 7, 8, 4, 3],
        "Electrochemical": [8, 6, 9, 8, 8, 5, 4],
        "Optical array": [8, 7, 7, 5, 6, 7, 6],
        "Ion mobility": [8, 7, 6, 4, 6, 7, 7],
        "Micro-GC": [8, 8, 5, 5, 5, 7, 8],
        "GC-MS": [9, 10, 2, 2, 3, 9, 10],
        "Biohybrid receptor": [8, 8, 5, 4, 5, 4, 7],
    }

    criteria_hw = ["Sensitivity", "Selectivity", "Portability", "Cost advantage", "Throughput", "Drift resistance", "Chemical insight"]

    chosen_platform = st.selectbox("Choose sensing platform", platforms)
    score_values = []

    cols = st.columns(4)
    for i, crit in enumerate(criteria_hw):
        with cols[i % 4]:
            score_values.append(st.slider(crit, 0, 10, default_scores[chosen_platform][i], key=f"hw_{crit}"))

    hw_df = pd.DataFrame({"Criterion": criteria_hw, "Score": score_values})
    readiness = float(np.mean(score_values))

    st.metric("Hardware readiness score", f"{readiness:.1f}/10")

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=hw_df["Score"], theta=hw_df["Criterion"], fill="toself", name=chosen_platform))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, title=f"Readiness radar: {chosen_platform}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        platform_table = pd.DataFrame(default_scores, index=criteria_hw).T.reset_index().rename(columns={"index": "Platform"})
        platform_table["Mean readiness"] = platform_table[criteria_hw].mean(axis=1)
        fig = px.bar(platform_table.sort_values("Mean readiness"), x="Mean readiness", y="Platform", orientation="h", title="Default platform comparison")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(platform_table, use_container_width=True)


st.markdown("---")
st.caption("OlfactoSense Studio, simulated data for research design and demonstration. Not a diagnostic, safety, or food-quality product.")
