# PredictMaint PY: Manutenção Preditiva com Machine Learning

Pipeline em Python que prevê se uma máquina industrial vai falhar, a partir de leituras de sensores, comparando os algoritmos **KNN** e **Árvore de Decisão** pela acurácia nos dados de teste.

Projeto avaliativo do Módulo 1 do curso Desenvolvedor(a) em IA para Análise Preditiva (Fundamentos de Dados, Programação e Análise Preditiva com Python).

## Problema que resolve

Paradas não planejadas na linha de produção geram custo e atraso. Com 10.000 leituras de sensores (temperatura do ar e do processo, velocidade de rotação, torque e desgaste da ferramenta), o projeto constrói um modelo que indica se a máquina vai falhar (`falha_maquina = 1`) ou operar normalmente (`0`), para que a equipe de manutenção priorize as inspeções.

## Resultado

| Modelo | Parâmetro | Acurácia no teste | Falhas detectadas (de 68) |
|---|---|---|---|
| KNN | K=1 | 91,85% | 35 (51%) |
| Árvore de Decisão | max_depth=None | **94,20%** | 48 (71%) |

**Modelo vencedor: Árvore de Decisão.** Como só 3,4% das máquinas falham, a acurácia de quem sempre prevê "normal" seria 96,6%, mas esse modelo nunca detectaria uma falha. Por isso o notebook também analisa matrizes de confusão e o recall da classe de falha.

## Etapas do pipeline

1. **Análise exploratória (EDA):** dimensões, tipos, estatísticas descritivas, distribuições, balanceamento da variável alvo e correlações.
2. **Limpeza e tratamento:** verificação de duplicados, imputação dos 2.000 valores nulos (mediana para a rotação, que é assimétrica, e média para as demais colunas) e análise de outliers por boxplot e critério do IQR.
3. **Feature engineering:** criação da variável `potencia = velocidade_rotacao_rpm * torque_nm`.
4. **Divisão e balanceamento:** divisão estratificada 80/20 e SMOTE aplicado somente no treino, sem vazamento de dados. As colunas de tipo de falha (`falha_*`) ficam fora do modelo.
5. **Escalonamento:** `StandardScaler` com `fit` apenas no treino, usado somente pelo KNN.
6. **Ajuste de parâmetros:** variação de `K` (KNN) e `max_depth` (árvore), comparando acurácia de treino e teste para identificar overfitting.
7. **Avaliação final:** acurácia no teste, matrizes de confusão, relatório por classe e veredito.

## Tecnologias

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn (KNN, Árvore de Decisão, StandardScaler, métricas) e imbalanced-learn (SMOTE). Execução em Jupyter Notebook.

## Estrutura do repositório

```
predictmaint-py/
├── data/manutencao_preditiva.csv       # base de dados (separador ponto e vírgula)
├── pipeline_manutencao_preditiva.ipynb # notebook com todo o pipeline
├── requirements.txt                    # dependências com versões fixas
├── KANBAN.md                           # quadro de tarefas do projeto
└── README.md
```

## Como executar

No Windows (PowerShell):

```powershell
git clone https://github.com/felipemendes2077/predictmaint-py.git
cd predictmaint-py
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m notebook
```

Se o PowerShell bloquear o `Activate.ps1`, rode antes `Set-ExecutionPolicy -Scope Process Bypass -Force`.

Em Linux ou macOS, ative o ambiente com `source venv/bin/activate`.

No navegador, abra `pipeline_manutencao_preditiva.ipynb` e use **Kernel > Restart Kernel and Run All Cells**.

## Melhorias futuras

- Escolher o modelo por recall, precisão e F1 da classe de falha, e não só pela acurácia.
- Validação cruzada estratificada e busca de hiperparâmetros (`GridSearchCV`).
- Testar Random Forest e Gradient Boosting e ajustar o limiar de decisão.
- Criar novas variáveis derivadas, como a relação entre desgaste e torque.
- Coletar mais exemplos de falha, hoje apenas 339 casos.

## Fluxo de trabalho

O projeto segue GitHub Flow: cada etapa foi desenvolvida em uma branch própria (`feature/...`) criada a partir da `develop`, integrada por Pull Request, e a versão final foi levada à `main`. O andamento das tarefas está em `KANBAN.md`.

## Autor

Felipe Mendes dos Santos