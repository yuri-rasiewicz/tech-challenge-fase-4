# Tech Challenge Fase 4 — Sistema Preditivo de Obesidade

Projeto desenvolvido para o Tech Challenge da Fase 4 da Pós Tech em Data Analytics.

A solução utiliza Machine Learning para prever o nível de obesidade de uma pessoa em sete classes, com base em características físicas, hábitos alimentares, histórico familiar e estilo de vida. A aplicação em Streamlit contém um dashboard analítico e um sistema preditivo.

## Estrutura do projeto

```text
tech-challenge-fase-4-obesidade/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── data/
│   ├── Obesity.csv
│   └── obesity_tratada.csv
│
├── models/
│   └── modelo_obesidade.pkl
│
├── notebooks/
│   └── Tech_Challenge_Fase_4_Obesidade_Final.ipynb
│
└── outputs/
    ├── comparacao_modelos_pt.csv
    ├── importancia_variaveis_pt.csv
    ├── relatorio_classificacao_pt.csv
    ├── validacao_overfitting.csv
    └── validacao_cruzada.csv
```

## Modelo final

O modelo selecionado foi o **Random Forest**, com os seguintes resultados no conjunto de teste:

- Acurácia: **98,09%**
- F1-score Macro: **98,04%**
- ROC AUC multiclasse: **0,9999**

A validação adicional contra overfitting comparou treino, teste e validação cruzada estratificada. O modelo apresentou alta performance no teste e resultados estáveis na validação cruzada.

## Como executar localmente

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o app:

```bash
streamlit run app.py
```

## Observação importante

O arquivo `modelo_obesidade.pkl` foi salvo usando `scikit-learn==1.6.1`. Por isso, o projeto fixa essa versão no arquivo `requirements.txt` para evitar problemas de compatibilidade ao carregar o modelo.

## Limites de uso e privacidade

- A base de treinamento contém idades entre **14 e 61 anos**. O app aceita idades de até 100 anos, mas resultados acima de 61 anos representam extrapolação e devem ser interpretados com maior cautela.
- Os dados informados no sistema preditivo são processados apenas durante a sessão. A aplicação não grava as informações em arquivo ou banco de dados.
- Este sistema foi desenvolvido para fins acadêmicos e funciona como ferramenta de apoio à decisão. Ele não substitui avaliação médica profissional.
