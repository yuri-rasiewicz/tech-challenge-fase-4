# ============================================================
# Tech Challenge Fase 4 - Sistema Preditivo de Obesidade
# Aplicação Streamlit com dashboard analítico e modelo preditivo
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------

st.set_page_config(
    page_title="Tech Challenge Fase 4 | Obesidade",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "obesity_tratada.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_obesidade.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "comparacao_modelos_pt.csv"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "importancia_variaveis_pt.csv"
OVERFIT_PATH = BASE_DIR / "outputs" / "validacao_overfitting.csv"
CV_PATH = BASE_DIR / "outputs" / "validacao_cruzada.csv"

# ------------------------------------------------------------
# Paleta visual
# ------------------------------------------------------------

TERRACOTA = "#C85A3B"
TERRACOTA_ESCURO = "#8C3B2E"
AREIA = "#F7EFE8"
AZUL_PETROLEO = "#2E5E6E"
DOURADO = "#E0A458"
VERDE_ACINZENTADO = "#6F8F7A"
CINZA_TEXTO = "#333333"

CORES_GRUPOS = {
    "Abaixo do Peso": "#6E8E9E",
    "Peso Normal": "#5A9E8F",
    "Sobrepeso": DOURADO,
    "Obesidade": TERRACOTA,
}

CORES_CLASSES = [
    "#6E8E9E",
    "#5A9E8F",
    "#A8B878",
    DOURADO,
    "#D97D54",
    TERRACOTA,
    TERRACOTA_ESCURO,
]

# ------------------------------------------------------------
# CSS leve para acabamento visual
# ------------------------------------------------------------

st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 1.8rem;
    }}
    [data-testid="stSidebar"] {{
        background-color: {AREIA};
    }}
    [data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #EFE1D8;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }}
    h1, h2, h3 {{
        color: {CINZA_TEXTO};
    }}
    .insight-card {{
        background-color: #FFF8F4;
        border-left: 5px solid {TERRACOTA};
        padding: 14px 18px;
        border-radius: 10px;
        margin: 12px 0px 18px 0px;
        font-size: 0.98rem;
    }}
    .soft-card {{
        background-color: #FFFFFF;
        border: 1px solid #EFE1D8;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 8px 0px 14px 0px;
    }}
    [data-testid="stSidebar"] div.stButton > button {{
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        border-radius: 10px;
        padding: 0.62rem 0.85rem;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Dicionários de tradução e ordem de exibição
# ------------------------------------------------------------

CLASSES_ORDEM = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III"
]

TRADUCAO_CLASSES = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}

MAPA_GRUPO = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso",
    "Overweight_Level_II": "Sobrepeso",
    "Obesity_Type_I": "Obesidade",
    "Obesity_Type_II": "Obesidade",
    "Obesity_Type_III": "Obesidade"
}

GRUPOS_ORDEM = ["Abaixo do Peso", "Peso Normal", "Sobrepeso", "Obesidade"]
CLASSES_PT_ORDEM = [TRADUCAO_CLASSES[c] for c in CLASSES_ORDEM]
SIM_NAO_PT = {"yes": "Sim", "no": "Não"}

FAF_LABELS = {
    0: "Nenhuma",
    1: "1 a 2x/semana",
    2: "3 a 4x/semana",
    3: "5x ou mais/semana"
}

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Garante que as colunas auxiliares existam mesmo se a base for substituída.
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    if "Obesity_Group" not in df.columns:
        df["Obesity_Group"] = df["Obesity"].map(MAPA_GRUPO)

    df["Classe"] = df["Obesity"].map(TRADUCAO_CLASSES)
    df["Grupo"] = df["Obesity"].map(MAPA_GRUPO)
    df["Histórico Familiar"] = df["family_history"].map(SIM_NAO_PT)
    df["Alimentos Calóricos"] = df["FAVC"].map(SIM_NAO_PT)
    df["Fuma"] = df["SMOKE"].map(SIM_NAO_PT)
    df["Monitora Calorias"] = df["SCC"].map(SIM_NAO_PT)

    # O dicionário informa que FAF é uma escala de 0 a 3. Para o dashboard,
    # arredondamos apenas para facilitar a leitura visual.
    df["Atividade Física"] = (
        df["FAF"].round().clip(0, 3).astype(int).map(FAF_LABELS)
    )

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        df_csv = pd.read_csv(path)
        if "Unnamed: 0" in df_csv.columns:
            df_csv = df_csv.rename(columns={"Unnamed: 0": "Classe/Métrica"})
        return df_csv
    return pd.DataFrame()


def percentual_por_categoria(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    tabela = pd.crosstab(df[coluna], df["Grupo"], normalize="index") * 100
    tabela = tabela.reindex(columns=GRUPOS_ORDEM)
    return tabela.round(2).reset_index().melt(
        id_vars=coluna,
        var_name="Grupo",
        value_name="Percentual"
    )


def percentual_atividade_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    ordem_atividade = list(FAF_LABELS.values())
    tabela = pd.crosstab(df["Grupo"], df["Atividade Física"], normalize="index") * 100
    tabela = tabela.reindex(index=GRUPOS_ORDEM, columns=ordem_atividade)
    return tabela.round(2).reset_index().melt(
        id_vars="Grupo",
        var_name="Atividade Física",
        value_name="Percentual"
    )


def traduzir_classe(classe: str) -> str:
    return TRADUCAO_CLASSES.get(str(classe), str(classe))


def grupo_simplificado(classe: str) -> str:
    return MAPA_GRUPO.get(str(classe), "Não identificado")


def formatar_percentual(valor: float) -> str:
    return f"{valor:.2f}%".replace(".", ",")


@st.cache_data
def calcular_matriz_confusao(df_base: pd.DataFrame) -> np.ndarray:
    modelo = load_model()
    colunas_excluir = ["Obesity", "Obesity_Group", "Classe", "Grupo", "Histórico Familiar",
                       "Alimentos Calóricos", "Fuma", "Monitora Calorias", "Atividade Física"]
    X = df_base.drop(columns=[c for c in colunas_excluir if c in df_base.columns])
    y = df_base["Obesity"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    y_pred = modelo.predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=CLASSES_ORDEM)

# ------------------------------------------------------------
# Carregamento dos dados
# ------------------------------------------------------------

df = load_data()
metricas_modelos = load_optional_csv(METRICS_PATH)
importancia_variaveis = load_optional_csv(IMPORTANCE_PATH)
validacao_overfit = load_optional_csv(OVERFIT_PATH)
validacao_cv = load_optional_csv(CV_PATH)

# ------------------------------------------------------------
# Menu lateral
# ------------------------------------------------------------

st.sidebar.title("Tech Challenge Fase 4")
st.sidebar.markdown("**Data Viz and Production Models**")

opcoes_paginas = [
    ("🏠 Contexto do Projeto", "Contexto do Projeto"),
    ("📊 Dashboard Analítico", "Dashboard Analítico"),
    ("🩺 Sistema Preditivo", "Sistema Preditivo"),
    ("🤖 Sobre o Modelo", "Sobre o Modelo")
]

if "pagina_atual" not in st.session_state:
    st.session_state.pagina_atual = "Contexto do Projeto"

st.sidebar.markdown("**Selecione a seção**")
for rotulo, valor in opcoes_paginas:
    selecionada = st.session_state.pagina_atual == valor
    if st.sidebar.button(
        rotulo,
        key=f"menu_{valor}",
        use_container_width=True,
        type="primary" if selecionada else "secondary"
    ):
        st.session_state.pagina_atual = valor
        st.rerun()

pagina = st.session_state.pagina_atual

st.sidebar.markdown("---")
st.sidebar.caption("Projeto acadêmico | Pós Tech Data Analytics")

# ------------------------------------------------------------
# Página 1 - Contexto
# ------------------------------------------------------------

if pagina == "Contexto do Projeto":
    st.title("Sistema Preditivo para Classificação do Nível de Obesidade")

    st.markdown(
        """
        Este projeto foi desenvolvido para o **Tech Challenge da Fase 4** da Pós Tech em Data Analytics.

        O objetivo é construir uma solução de apoio à decisão para estimar o nível de obesidade de uma pessoa
        a partir de características físicas, hábitos alimentares, histórico familiar e estilo de vida.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros analisados", f"{len(df):,}".replace(",", "."))
    col2.metric("Classes previstas", "7")
    col3.metric("Modelo final", "Random Forest")
    col4.metric("Acurácia", "98,09%")

    st.subheader("Abordagem adotada")
    st.markdown(
        """
        - O problema foi tratado como uma **classificação multiclasse**, mantendo as sete classes originais da base.
        - Para facilitar a leitura executiva, o dashboard também apresenta uma visão simplificada em quatro grupos:
          **Abaixo do Peso, Peso Normal, Sobrepeso e Obesidade**.
        - A variável **IMC** foi criada como etapa de *feature engineering*, a partir da relação entre peso e altura.
        - A pipeline de Machine Learning foi salva e integrada ao sistema preditivo em Streamlit.
        """
    )

    st.markdown(
        """
        <div class="insight-card">
        <b>Mensagem principal:</b> a solução combina análise exploratória, modelo preditivo e visualização executiva
        em uma única aplicação. O objetivo não é substituir o diagnóstico médico, mas oferecer um apoio inicial
        baseado nos padrões encontrados na base de dados.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Este sistema é uma ferramenta acadêmica de apoio à decisão e não substitui avaliação médica profissional."
    )

# ------------------------------------------------------------
# Página 2 - Dashboard
# ------------------------------------------------------------

elif pagina == "Dashboard Analítico":
    st.title("Dashboard Analítico")
    st.markdown(
        """
        A leitura do dashboard segue uma linha simples: primeiro entendemos a distribuição geral da base,
        depois analisamos o perfil corporal e, por fim, observamos hábitos e fatores de estilo de vida
        associados aos grupos de obesidade.
        """
    )

    aba1, aba2, aba3 = st.tabs([
        "1. Visão Geral",
        "2. Perfil Corporal",
        "3. Hábitos e Estilo de Vida"
    ])

    with aba1:
        st.markdown(
            """
            <div class="insight-card">
            <b>1. Visão geral:</b> antes de interpretar padrões, é importante verificar se a base está distribuída
            entre as classes. Isso ajuda a entender se o modelo não está aprendendo apenas uma classe dominante.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1.2, 0.8])

        distribuicao_classes = (
            df["Classe"]
            .value_counts()
            .reindex(CLASSES_PT_ORDEM)
            .reset_index()
        )
        distribuicao_classes.columns = ["Classe", "Quantidade"]

        fig_classes = px.bar(
            distribuicao_classes,
            x="Classe",
            y="Quantidade",
            title="Distribuição dos Níveis de Obesidade",
            text="Quantidade",
            category_orders={"Classe": CLASSES_PT_ORDEM},
            color="Classe",
            color_discrete_sequence=CORES_CLASSES
        )
        fig_classes.update_layout(
            xaxis_title="Nível de obesidade",
            yaxis_title="Quantidade de registros",
            showlegend=False
        )
        col1.plotly_chart(fig_classes, use_container_width=True)

        distribuicao_grupos = (
            df["Grupo"]
            .value_counts()
            .reindex(GRUPOS_ORDEM)
            .reset_index()
        )
        distribuicao_grupos.columns = ["Grupo", "Quantidade"]

        fig_grupos = px.pie(
            distribuicao_grupos,
            names="Grupo",
            values="Quantidade",
            title="Visão Simplificada dos Grupos",
            hole=0.45,
            color="Grupo",
            color_discrete_map=CORES_GRUPOS
        )
        fig_grupos.update_traces(textposition="inside", textinfo="percent+label")
        col2.plotly_chart(fig_grupos, use_container_width=True)

        st.markdown(
            """
            **Leitura executiva:** a base possui distribuição relativamente equilibrada entre as sete classes.
            Na visão agrupada, o maior volume está no grupo de obesidade, seguido por sobrepeso. Essa visão
            simplificada facilita a comunicação para uma audiência de negócio, enquanto o modelo preserva as sete
            classes originais.
            """
        )

    with aba2:
        st.markdown(
            """
            <div class="insight-card">
            <b>2. Perfil corporal:</b> peso, altura e IMC são variáveis centrais para entender o nível de obesidade.
            O IMC é um forte indicador inicial, mas não deve ser interpretado como diagnóstico isolado, pois uma
            avaliação clínica também considera composição corporal, histórico, exames e contexto individual.
            </div>
            """,
            unsafe_allow_html=True
        )

        imc_medio = (
            df.groupby("Classe", observed=True)["BMI"]
            .mean()
            .reindex(CLASSES_PT_ORDEM)
            .round(2)
            .reset_index()
        )
        imc_medio.columns = ["Classe", "IMC Médio"]

        fig_imc = px.line(
            imc_medio,
            x="Classe",
            y="IMC Médio",
            title="Evolução do IMC Médio por Nível de Obesidade",
            markers=True,
            text="IMC Médio",
            category_orders={"Classe": CLASSES_PT_ORDEM},
            color_discrete_sequence=[TERRACOTA]
        )
        fig_imc.update_traces(textposition="top center", line=dict(width=3))
        fig_imc.update_layout(xaxis_title="Nível de obesidade", yaxis_title="IMC médio")
        st.plotly_chart(fig_imc, use_container_width=True)

        col1, col2 = st.columns(2)

        fig_box = px.box(
            df,
            x="Classe",
            y="BMI",
            title="Distribuição do IMC por Nível de Obesidade",
            category_orders={"Classe": CLASSES_PT_ORDEM},
            color="Classe",
            color_discrete_sequence=CORES_CLASSES
        )
        fig_box.update_layout(
            xaxis_title="Nível de obesidade",
            yaxis_title="IMC",
            showlegend=False
        )
        col1.plotly_chart(fig_box, use_container_width=True)

        peso_medio = (
            df.groupby("Classe", observed=True)["Weight"]
            .mean()
            .reindex(CLASSES_PT_ORDEM)
            .round(2)
            .reset_index()
        )
        peso_medio.columns = ["Classe", "Peso Médio"]

        fig_peso = px.bar(
            peso_medio,
            x="Peso Médio",
            y="Classe",
            title="Peso Médio por Nível de Obesidade",
            text="Peso Médio",
            category_orders={"Classe": CLASSES_PT_ORDEM},
            color="Classe",
            color_discrete_sequence=CORES_CLASSES
        )
        fig_peso.update_layout(
            xaxis_title="Peso médio em kg",
            yaxis_title="Nível de obesidade",
            showlegend=False
        )
        col2.plotly_chart(fig_peso, use_container_width=True)

        st.markdown(
            """
            **Leitura executiva:** o IMC médio cresce de forma consistente conforme avançamos para classes mais
            altas de obesidade. Isso explica por que o IMC se tornou a variável mais importante do modelo. Ainda
            assim, o projeto mantém variáveis comportamentais e de estilo de vida para complementar a análise.
            """
        )

    with aba3:
        st.markdown(
            """
            <div class="insight-card">
            <b>3. Hábitos e estilo de vida:</b> esta etapa analisa três dimensões complementares: predisposição
            familiar, frequência de atividade física e meio de transporte principal. Os gráficos mostram como os
            grupos corporais se distribuem dentro de cada categoria observada na base.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Os percentuais descrevem associações observadas na base. Eles não demonstram relação de causa e efeito "
            "e não devem ser interpretados isoladamente como diagnóstico ou recomendação clínica."
        )

        df_habitos = df.copy()

        # ----------------------------------------------------
        # 1. Histórico familiar
        # ----------------------------------------------------
        st.subheader("1. Histórico familiar e estado corporal")
        st.markdown(
            "Entre quem respondeu **Sim** ou **Não** para histórico familiar de excesso de peso, o gráfico mostra "
            "a proporção de pessoas em cada grupo corporal."
        )

        ordem_historico = ["Não", "Sim"]
        tabela_historico = (
            pd.crosstab(
                df_habitos["Histórico Familiar"],
                df_habitos["Grupo"],
                normalize="index"
            )
            .mul(100)
            .reindex(index=ordem_historico, columns=GRUPOS_ORDEM)
            .round(1)
            .reset_index()
            .melt(
                id_vars="Histórico Familiar",
                var_name="Grupo corporal",
                value_name="Percentual"
            )
        )
        tabela_historico["Rótulo"] = tabela_historico["Percentual"].map(lambda x: f"{x:.1f}%")

        fig_historico = px.bar(
            tabela_historico,
            x="Histórico Familiar",
            y="Percentual",
            color="Grupo corporal",
            barmode="group",
            text="Rótulo",
            title="Distribuição do Estado Corporal por Histórico Familiar",
            category_orders={
                "Histórico Familiar": ordem_historico,
                "Grupo corporal": GRUPOS_ORDEM
            },
            color_discrete_map=CORES_GRUPOS
        )
        fig_historico.update_traces(
            textposition="outside",
            textfont_size=13,
            cliponaxis=False
        )
        fig_historico.update_layout(
            xaxis_title="Histórico familiar de excesso de peso",
            yaxis_title="Percentual dentro de cada resposta (%)",
            yaxis_range=[0, 65],
            legend_title="Grupo corporal",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            height=500,
            margin=dict(l=20, r=30, t=110, b=45)
        )
        st.plotly_chart(fig_historico, use_container_width=True)

        historico_sim = tabela_historico[
            tabela_historico["Histórico Familiar"] == "Sim"
        ].set_index("Grupo corporal")["Percentual"]
        historico_nao = tabela_historico[
            tabela_historico["Histórico Familiar"] == "Não"
        ].set_index("Grupo corporal")["Percentual"]
        excesso_sim = historico_sim.get("Sobrepeso", 0) + historico_sim.get("Obesidade", 0)
        excesso_nao = historico_nao.get("Sobrepeso", 0) + historico_nao.get("Obesidade", 0)

        st.markdown(
            f"""
            **Leitura:** entre pessoas com histórico familiar, **{excesso_sim:.1f}%** estão em sobrepeso ou obesidade,
            contra **{excesso_nao:.1f}%** entre quem não relatou esse histórico. A diferença indica uma associação
            relevante na base, mas não prova que o histórico familiar, sozinho, determine o estado corporal.
            """
        )

        st.divider()

        # ----------------------------------------------------
        # 2. Frequência de atividade física
        # ----------------------------------------------------
        st.subheader("2. Frequência de atividade física")
        st.markdown(
            "A variável de atividade física utiliza quatro faixas: nenhuma, 1 a 2 vezes por semana, 3 a 4 vezes por "
            "semana e 5 vezes ou mais. Para cada faixa, o gráfico apresenta separadamente abaixo do peso, peso "
            "normal, sobrepeso e obesidade."
        )

        ordem_atividade = ["Nenhuma", "1 a 2x/semana", "3 a 4x/semana", "5x ou mais/semana"]
        ordem_resultado_atividade = GRUPOS_ORDEM

        df_habitos["Resultado corporal"] = df_habitos["Grupo"]

        tabela_atividade = (
            pd.crosstab(
                df_habitos["Atividade Física"],
                df_habitos["Resultado corporal"],
                normalize="index"
            )
            .mul(100)
            .reindex(index=ordem_atividade, columns=ordem_resultado_atividade)
            .round(1)
            .reset_index()
            .melt(
                id_vars="Atividade Física",
                var_name="Resultado corporal",
                value_name="Percentual"
            )
        )
        tabela_atividade["Rótulo"] = tabela_atividade["Percentual"].map(lambda x: f"{x:.1f}%")

        fig_atividade = px.bar(
            tabela_atividade,
            x="Atividade Física",
            y="Percentual",
            color="Resultado corporal",
            barmode="group",
            text="Rótulo",
            title="Estado Corporal por Frequência de Atividade Física",
            category_orders={
                "Atividade Física": ordem_atividade,
                "Resultado corporal": ordem_resultado_atividade
            },
            color_discrete_map=CORES_GRUPOS
        )
        fig_atividade.update_traces(
            textposition="outside",
            textfont_size=12,
            cliponaxis=False
        )
        fig_atividade.update_layout(
            xaxis_title="Frequência de atividade física",
            yaxis_title="Percentual dentro de cada faixa (%)",
            yaxis_range=[0, 65],
            legend_title="Grupo corporal",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            height=540,
            margin=dict(l=20, r=30, t=110, b=55)
        )
        st.plotly_chart(fig_atividade, use_container_width=True)

        atividade_obesidade = tabela_atividade[
            tabela_atividade["Resultado corporal"] == "Obesidade"
        ].set_index("Atividade Física")["Percentual"]
        atividade_normal = tabela_atividade[
            tabela_atividade["Resultado corporal"] == "Peso Normal"
        ].set_index("Atividade Física")["Percentual"]

        st.markdown(
            f"""
            **Leitura:** a proporção de obesidade cai de **{atividade_obesidade.get('Nenhuma', 0):.1f}%** entre
            pessoas sem atividade física para **{atividade_obesidade.get('5x ou mais/semana', 0):.1f}%** entre as
            que praticam cinco vezes ou mais por semana. No mesmo intervalo, o peso normal passa de
            **{atividade_normal.get('Nenhuma', 0):.1f}%** para **{atividade_normal.get('5x ou mais/semana', 0):.1f}%**.
            O padrão sugere associação entre maior frequência de exercício e melhor estado corporal na base.
            """
        )
        st.caption(
            "As quatro classificações são apresentadas separadamente para evitar que situações diferentes, como "
            "abaixo do peso e sobrepeso, sejam reunidas em um mesmo grupo genérico."
        )

        st.divider()

        # ----------------------------------------------------
        # 3. Meio de transporte principal
        # ----------------------------------------------------
        st.subheader("3. Meio de transporte principal")
        st.markdown(
            "O meio de transporte pode funcionar como um indicador indireto da rotina de deslocamento. O gráfico "
            "compara, dentro de cada categoria, a proporção de pessoas abaixo do peso, com peso normal e com excesso "
            "de peso."
        )

        mapa_transporte = {
            "Automobile": "Automóvel",
            "Motorbike": "Moto",
            "Bike": "Bicicleta",
            "Public_Transportation": "Transporte público",
            "Walking": "Caminhada"
        }
        ordem_transporte = ["Automóvel", "Transporte público", "Caminhada", "Moto", "Bicicleta"]
        ordem_situacao = ["Abaixo do peso", "Peso normal", "Excesso de peso"]

        df_habitos["Meio de Transporte"] = df_habitos["MTRANS"].map(mapa_transporte)
        df_habitos["Situação corporal"] = np.select(
            [
                df_habitos["Grupo"].eq("Abaixo do Peso"),
                df_habitos["Grupo"].eq("Peso Normal")
            ],
            ["Abaixo do peso", "Peso normal"],
            default="Excesso de peso"
        )

        contagem_transporte = df_habitos["Meio de Transporte"].value_counts()
        rotulo_transporte = {
            transporte: f"{transporte} (n={int(contagem_transporte.get(transporte, 0))})"
            for transporte in ordem_transporte
        }
        df_habitos["Transporte com amostra"] = df_habitos["Meio de Transporte"].map(rotulo_transporte)
        ordem_transporte_rotulo = [rotulo_transporte[t] for t in ordem_transporte]

        tabela_transporte = (
            pd.crosstab(
                df_habitos["Transporte com amostra"],
                df_habitos["Situação corporal"],
                normalize="index"
            )
            .mul(100)
            .reindex(index=ordem_transporte_rotulo, columns=ordem_situacao)
            .round(1)
            .reset_index()
            .melt(
                id_vars="Transporte com amostra",
                var_name="Situação corporal",
                value_name="Percentual"
            )
        )
        tabela_transporte["Rótulo"] = tabela_transporte["Percentual"].map(lambda x: f"{x:.1f}%")

        fig_transporte = px.bar(
            tabela_transporte,
            x="Percentual",
            y="Transporte com amostra",
            color="Situação corporal",
            barmode="group",
            orientation="h",
            text="Rótulo",
            title="Estado Corporal por Meio de Transporte Principal",
            category_orders={
                "Transporte com amostra": ordem_transporte_rotulo,
                "Situação corporal": ordem_situacao
            },
            color_discrete_map={
                "Abaixo do peso": AZUL_PETROLEO,
                "Peso normal": VERDE_ACINZENTADO,
                "Excesso de peso": TERRACOTA
            }
        )
        fig_transporte.update_traces(
            textposition="outside",
            textfont_size=12,
            cliponaxis=False
        )
        fig_transporte.update_layout(
            xaxis_title="Percentual dentro de cada meio de transporte (%)",
            yaxis_title=None,
            xaxis_range=[0, 90],
            legend_title="Situação corporal",
            height=560,
            margin=dict(l=20, r=75, t=70, b=45)
        )
        fig_transporte.update_yaxes(
            categoryorder="array",
            categoryarray=ordem_transporte_rotulo,
            autorange="reversed"
        )
        st.plotly_chart(fig_transporte, use_container_width=True)

        tabela_transporte_base = tabela_transporte.copy()
        tabela_transporte_base["Meio de Transporte"] = tabela_transporte_base["Transporte com amostra"].str.replace(
            r" \(n=\d+\)$", "", regex=True
        )
        # Para a leitura executiva, priorizamos categorias com pelo menos 30 registros.
        # As categorias menores continuam visíveis no gráfico, mas não são usadas para generalizações.
        transportes_amostra_adequada = contagem_transporte[contagem_transporte >= 30].index.tolist()
        normal_transporte = tabela_transporte_base[
            (tabela_transporte_base["Situação corporal"] == "Peso normal")
            & (tabela_transporte_base["Meio de Transporte"].isin(transportes_amostra_adequada))
        ].sort_values("Percentual", ascending=False)
        excesso_transporte = tabela_transporte_base[
            (tabela_transporte_base["Situação corporal"] == "Excesso de peso")
            & (tabela_transporte_base["Meio de Transporte"].isin(transportes_amostra_adequada))
        ].sort_values("Percentual", ascending=False)

        melhor_normal = normal_transporte.iloc[0]
        maior_excesso = excesso_transporte.iloc[0]

        st.markdown(
            f"""
            **Leitura:** considerando os meios com pelo menos 30 registros, **{melhor_normal['Meio de Transporte']}**
            apresenta a maior proporção de peso normal ({melhor_normal['Percentual']:.1f}%), enquanto
            **{maior_excesso['Meio de Transporte']}** apresenta a maior proporção de excesso de peso
            ({maior_excesso['Percentual']:.1f}%). O resultado sugere associação com a rotina de deslocamento, mas não
            permite concluir que o meio de transporte seja a causa do estado corporal.
            """
        )
        st.warning(
            "Bicicleta e moto possuem poucos registros na base. Por isso, seus percentuais são mais sensíveis a "
            "variações e não devem ser generalizados isoladamente."
        )

        st.markdown(
            """
            **Síntese da análise:** o histórico familiar apresenta a associação mais forte com excesso de peso; a
            frequência de atividade física mostra um padrão favorável conforme aumenta; e o meio de transporte sugere
            diferenças relacionadas à rotina de deslocamento, embora algumas categorias tenham amostras pequenas.
            Embora os gráficos analisem cada fator separadamente, o modelo preditivo avalia essas características em
            conjunto com as demais variáveis disponíveis.
            """
        )

# ------------------------------------------------------------
# Página 3 - Sistema Preditivo
# ------------------------------------------------------------

elif pagina == "Sistema Preditivo":
    st.title("Sistema Preditivo")
    st.markdown(
        "Preencha as informações abaixo para estimar o nível de obesidade previsto pelo modelo."
    )

    try:
        modelo = load_model()
    except Exception as erro:
        st.error("Não foi possível carregar o modelo preditivo.")
        st.exception(erro)
        st.stop()

    with st.form("form_predicao"):
        st.subheader("Dados físicos")
        col1, col2, col3, col4 = st.columns(4)

        genero_pt = col1.selectbox("Gênero", ["Feminino", "Masculino"])
        idade = col2.number_input(
            "Idade",
            min_value=14.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            help="A base de treinamento contém idades entre 14 e 61 anos. Valores acima de 61 são aceitos, mas representam extrapolação e exigem maior cautela na interpretação."
        )
        altura = col3.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01)
        peso = col4.number_input("Peso (kg)", min_value=35.0, max_value=180.0, value=75.0, step=0.5)

        bmi = peso / (altura ** 2)
        st.caption(f"IMC calculado automaticamente: **{bmi:.2f}**")

        st.subheader("Histórico, alimentação e estilo de vida")
        col1, col2, col3 = st.columns(3)

        historico_pt = col1.selectbox("Histórico familiar de excesso de peso?", ["Sim", "Não"])
        alimentos_caloricos_pt = col2.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
        vegetais_pt = col3.selectbox("Consumo de vegetais", ["Raramente", "Às vezes", "Sempre"])

        col1, col2, col3 = st.columns(3)
        refeicoes_pt = col1.selectbox("Refeições principais por dia", ["1", "2", "3", "4 ou mais"])
        lanches_pt = col2.selectbox("Come entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        fuma_pt = col3.selectbox("Fuma?", ["Não", "Sim"])

        col1, col2, col3 = st.columns(3)
        agua_pt = col1.selectbox("Consumo diário de água", ["Menos de 1L", "1L a 2L", "Mais de 2L"])
        monitora_calorias_pt = col2.selectbox("Monitora calorias?", ["Não", "Sim"])
        atividade_pt = col3.selectbox("Atividade física", ["Nenhuma", "1 a 2x/semana", "3 a 4x/semana", "5x ou mais/semana"])

        col1, col2, col3 = st.columns(3)
        tecnologia_pt = col1.selectbox("Uso diário de dispositivos eletrônicos", ["0 a 2h/dia", "3 a 5h/dia", "Mais de 5h/dia"])
        alcool_pt = col2.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        transporte_pt = col3.selectbox("Meio de transporte principal", ["Automóvel", "Moto", "Bicicleta", "Transporte público", "Caminhada"])

        submit = st.form_submit_button("Gerar previsão")

    st.caption(
        "Privacidade: os dados informados são usados somente para gerar a previsão durante a sessão atual. "
        "Esta aplicação não grava as informações em arquivo ou banco de dados."
    )

    if submit:
        if idade > 61:
            st.warning(
                "A idade informada está acima do intervalo observado na base de treinamento (14 a 61 anos). "
                "A previsão será gerada, mas deve ser interpretada com cautela por representar extrapolação do modelo."
            )
        entrada = pd.DataFrame([{
            "Gender": {"Feminino": "Female", "Masculino": "Male"}[genero_pt],
            "Age": idade,
            "Height": altura,
            "Weight": peso,
            "family_history": {"Sim": "yes", "Não": "no"}[historico_pt],
            "FAVC": {"Sim": "yes", "Não": "no"}[alimentos_caloricos_pt],
            "FCVC": {"Raramente": 1, "Às vezes": 2, "Sempre": 3}[vegetais_pt],
            "NCP": {"1": 1, "2": 2, "3": 3, "4 ou mais": 4}[refeicoes_pt],
            "CAEC": {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}[lanches_pt],
            "SMOKE": {"Sim": "yes", "Não": "no"}[fuma_pt],
            "CH2O": {"Menos de 1L": 1, "1L a 2L": 2, "Mais de 2L": 3}[agua_pt],
            "SCC": {"Sim": "yes", "Não": "no"}[monitora_calorias_pt],
            "FAF": {"Nenhuma": 0, "1 a 2x/semana": 1, "3 a 4x/semana": 2, "5x ou mais/semana": 3}[atividade_pt],
            "TUE": {"0 a 2h/dia": 0, "3 a 5h/dia": 1, "Mais de 5h/dia": 2}[tecnologia_pt],
            "CALC": {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}[alcool_pt],
            "MTRANS": {
                "Automóvel": "Automobile",
                "Moto": "Motorbike",
                "Bicicleta": "Bike",
                "Transporte público": "Public_Transportation",
                "Caminhada": "Walking"
            }[transporte_pt],
            "BMI": bmi
        }])

        predicao = modelo.predict(entrada)[0]
        predicao_pt = traduzir_classe(predicao)
        grupo_pt = grupo_simplificado(predicao)

        st.success(f"Classificação prevista: **{predicao_pt}**")
        st.info(f"Grupo simplificado: **{grupo_pt}**")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]
            classes_modelo = getattr(modelo, "classes_", None)

            if classes_modelo is None and hasattr(modelo, "named_steps"):
                classes_modelo = modelo.named_steps["modelo"].classes_

            prob_df = pd.DataFrame({
                "Classe": [traduzir_classe(classe) for classe in classes_modelo],
                "Apoio do modelo (%)": probabilidades * 100
            }).sort_values("Apoio do modelo (%)", ascending=False)

            prob_principal = float(prob_df.iloc[0]["Apoio do modelo (%)"])

            with st.expander("Ver distribuição do apoio do modelo por classe"):
                st.markdown(
                    f"""
                    **Como interpretar:** o Random Forest distribui seu apoio interno entre as sete classes e escolhe
                    a classe com o maior valor. Neste resultado, **{predicao_pt}** recebeu
                    **{formatar_percentual(prob_principal)}** de apoio interno.

                    Esse percentual **não representa a chance clínica de o diagnóstico estar correto**. Ele mostra
                    apenas como o conjunto de árvores do modelo se dividiu entre as classes possíveis. Quando os
                    valores das primeiras classes ficam próximos, o perfil informado está perto de mais de uma
                    categoria e o resultado deve ser analisado com maior cautela.
                    """
                )

                prob_plot = prob_df.sort_values("Apoio do modelo (%)", ascending=True).copy()
                prob_plot["Rótulo"] = prob_plot["Apoio do modelo (%)"].map(lambda x: f"{x:.1f}%")
                fig_prob = px.bar(
                    prob_plot,
                    x="Apoio do modelo (%)",
                    y="Classe",
                    orientation="h",
                    title="Distribuição do Apoio Interno por Classe",
                    text="Rótulo",
                    color_discrete_sequence=[TERRACOTA]
                )
                fig_prob.update_traces(textposition="outside", textfont_size=13, cliponaxis=False)
                fig_prob.update_layout(
                    xaxis_title="Apoio interno do modelo (%)",
                    yaxis_title="Classe",
                    xaxis_range=[0, 100]
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        st.warning(
            "Este resultado é uma estimativa gerada por modelo de Machine Learning e não substitui "
            "avaliação clínica individualizada."
        )

# ------------------------------------------------------------
# Página 4 - Sobre o Modelo
# ------------------------------------------------------------

elif pagina == "Sobre o Modelo":
    st.title("Sobre o Modelo")

    st.markdown(
        """
        A solução foi construída como uma pipeline de Machine Learning com pré-processamento das variáveis,
        codificação de atributos categóricos, padronização das variáveis numéricas e treinamento do modelo final.
        """
    )

    st.markdown(
        """
        <div class="insight-card">
        <b>Modelo escolhido:</b> foram avaliados três algoritmos de classificação — Logistic Regression,
        Decision Tree e Random Forest. O Random Forest foi selecionado por apresentar a melhor performance geral,
        com acurácia de aproximadamente 98% e F1-score macro de aproximadamente 98% no conjunto de teste.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not metricas_modelos.empty:
        rf = metricas_modelos[metricas_modelos["Modelo"] == "Random Forest"].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Modelo final", "Random Forest")
        col2.metric("Acurácia no teste", f"{rf['Acurácia']:.2f}%".replace(".", ","))
        col3.metric("F1-score macro", f"{rf['F1-score Macro']:.2f}%".replace(".", ","))

    st.subheader("Comparação dos modelos")
    if not metricas_modelos.empty:
        st.dataframe(metricas_modelos, use_container_width=True)

        metricas_plot = metricas_modelos.sort_values("Acurácia", ascending=True)
        fig_metricas = px.bar(
            metricas_plot,
            x="Acurácia",
            y="Modelo",
            orientation="h",
            title="Acurácia dos Modelos Testados",
            text="Acurácia",
            color="Modelo",
            color_discrete_sequence=[AZUL_PETROLEO, DOURADO, TERRACOTA]
        )
        fig_metricas.update_layout(xaxis_title="Acurácia (%)", yaxis_title="Modelo", showlegend=False)
        st.plotly_chart(fig_metricas, use_container_width=True)

        st.markdown(
            """
            **Leitura:** os três modelos ficaram acima do requisito mínimo de 75% de assertividade.
            O Random Forest apresentou o melhor equilíbrio entre desempenho, robustez e capacidade de explicação
            por importância das variáveis.
            """
        )

    st.subheader("Matriz de confusão do Random Forest")
    matriz_confusao = calcular_matriz_confusao(df)
    fig_confusao = px.imshow(
        matriz_confusao,
        x=CLASSES_PT_ORDEM,
        y=CLASSES_PT_ORDEM,
        text_auto=True,
        color_continuous_scale="Oranges",
        title="Matriz de Confusão - Conjunto de Teste"
    )
    fig_confusao.update_layout(
        xaxis_title="Classe prevista pelo modelo",
        yaxis_title="Classe real"
    )
    st.plotly_chart(fig_confusao, use_container_width=True)
    st.markdown(
        """
        **Como ler:** a diagonal principal representa os acertos do modelo. Quanto mais concentrados os valores
        estiverem nessa diagonal, melhor o desempenho. Nesta matriz, a maior parte das previsões ficou na diagonal,
        e os poucos erros ocorreram principalmente entre classes próximas, como peso normal e sobrepeso ou tipos
        vizinhos de obesidade.
        """
    )

    st.subheader("Validação do modelo")
    if not validacao_overfit.empty and not validacao_cv.empty:
        teste_acc = float(validacao_overfit.loc[validacao_overfit["Base"] == "Teste", "Acurácia"].iloc[0])
        cv_acc = float(validacao_cv.loc[validacao_cv["Métrica"] == "Acurácia", "Teste CV - Média"].iloc[0])
        cv_desvio = float(validacao_cv.loc[validacao_cv["Métrica"] == "Acurácia", "Teste CV - Desvio"].iloc[0])

        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia no teste", f"{teste_acc:.2f}%".replace(".", ","))
        col2.metric("Média na validação cruzada", f"{cv_acc:.2f}%".replace(".", ","))
        col3.metric("Desvio entre as divisões", f"{cv_desvio:.2f} p.p.".replace(".", ","))

        st.info(
            "O modelo foi comparado entre treino e teste e também avaliado em cinco divisões estratificadas da base. "
            "O desempenho permaneceu próximo de 98% e apresentou baixa variação, sem sinais relevantes de overfitting."
        )

        with st.expander("Ver resultados detalhados da validação"):
            col1, col2 = st.columns(2)
            col1.markdown("**Treino vs Teste**")
            col1.dataframe(validacao_overfit, use_container_width=True)
            col2.markdown("**Validação Cruzada Estratificada**")
            col2.dataframe(validacao_cv, use_container_width=True)

    st.subheader("Importância das variáveis")
    if not importancia_variaveis.empty:
        if "importancia_percentual" in importancia_variaveis.columns:
            top_importancia = importancia_variaveis.head(15).copy()
            top_importancia["Rótulo"] = top_importancia["importancia_percentual"].map(lambda x: f"{x:.1f}%")
            top_importancia = top_importancia.sort_values("importancia_percentual", ascending=True)
            max_importancia = float(top_importancia["importancia_percentual"].max())
            fig_importancia = px.bar(
                top_importancia,
                x="importancia_percentual",
                y="variavel_traduzida",
                orientation="h",
                title="Top 15 Variáveis Mais Importantes",
                text="Rótulo",
                color_discrete_sequence=[TERRACOTA]
            )
            fig_importancia.update_traces(textposition="outside", textfont_size=15, cliponaxis=False)
            fig_importancia.update_layout(
                xaxis_title="Importância (%)",
                yaxis_title="Variável",
                xaxis_range=[0, max_importancia * 1.22],
                height=680,
                margin=dict(r=80)
            )
            st.plotly_chart(fig_importancia, use_container_width=True)
            st.markdown(
                """
                **Leitura:** o IMC e o peso foram as variáveis mais relevantes para a classificação, o que é
                coerente com o problema analisado. Variáveis comportamentais, como consumo de vegetais,
                refeições, uso de tecnologia e atividade física, aparecem com menor peso, mas ajudam a complementar
                a leitura do perfil do paciente.
                """
            )

    st.markdown(
        """
        **Observação:** a alta performance é coerente com a natureza do problema, pois a base possui variáveis
        fortemente relacionadas à classificação corporal, como peso, altura e IMC. O IMC foi criado como etapa
        de *feature engineering* e se mostrou a variável mais relevante para o modelo. Ainda assim, o resultado
        deve ser interpretado como apoio analítico, não como diagnóstico médico isolado.
        """
    )
