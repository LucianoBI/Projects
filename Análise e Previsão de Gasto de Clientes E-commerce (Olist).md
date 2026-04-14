<h1 align="center">
<b>📊 Análise e Previsão de Gasto de Clientes E-commerce (Olist)</b>
</h1>



<div align="center">
  <img width="800" height="1200" alt="image" src="https://github.com/user-attachments/assets/4c00b5f7-fece-4320-a10e-0cfb7ed8f955"/>
</div>




# 📚Visão Geral do Projeto

Este projeto foca na análise de dados transacionais de clientes de e-commerce da Olist, uma das maiores plataformas de marketplace do Brasil. 
O objetivo principal é compreender o comportamento de compra dos clientes, identificar padrões e desenvolver um modelo de Machine Learning para prever o valor total que um cliente pode gastar.

<p><p><p><p><p><p><p><p>

2. Exploração inicial dos dados

```python
df_customers.info()
df_customers.describe()
df_customers.isnull().sum()

---------------------------------------------------------
Data columns (total 5 columns):
 #   Column                    Non-Null Count  Dtype 
---  ------                    --------------  ----- 
 0   customer_id               99441 non-null  object
 1   customer_unique_id        99441 non-null  object
 2   customer_zip_code_prefix  99441 non-null  int64 
 3   customer_city             99441 non-null  object
 4   customer_state            99441 non-null  object
dtypes: int64(1), object(4)
memory usage: 3.8+ MB
0
customer_id	0
customer_unique_id	0
customer_zip_code_prefix	0

```

2. Limpeza e transformação de dados

```python

df_customers['customer_city'] = df_customers['customer_city'].str.lower().str.strip()
df_customers['customer_state'] = df_customers['customer_state'].str.upper()
customer_city	0
customer_state	0

```

3. Unindo os datasets


```python


df_orders_customers = df_orders.merge(df_customers, on="customer_id", how="inner")

df_orders_payments = df_orders_customers.merge(df_payments, on="order_id", how="inner")

df_full = df_orders_payments.merge(df_items, on="order_id", how="inner")

df_full = df_full.merge(df_products[["product_id","product_category_name"]],
                        on="product_id", how="left")

df_full.head()

dtype: int64

-------------------------------------------------------------------------------------------

order_id	customer_id	order_status	order_purchase_timestamp	order_approved_at	order_delivered_carrier_date	order_delivered_customer_date	order_estimated_delivery_date	customer_unique_id	customer_zip_code_prefix	...	payment_type	payment_installments	payment_value	order_item_id	product_id	seller_id	shipping_limit_date	price	freight_value	product_category_name
0	e481f51cbdc54678b7cc49136f2d6af7	9ef432eb6251297304e76186b10a928d	delivered	2017-10-02 10:56:33	2017-10-02 11:07:15	2017-10-04 19:55:00	2017-10-10 21:25:13	2017-10-18 00:00:00	7c396fd4830fd04220f754e42b4e5bff	3149	...	credit_card	1	18.12	1	87285b34884572647811a353c7ac498a	3504c0cb71d7fa48d967e0e4c94d59d9	2017-10-06 11:07:15	29.99	8.72	utilidades_domesticas
1	e481f51cbdc54678b7cc49136f2d6af7	9ef432eb6251297304e76186b10a928d	delivered	2017-10-02 10:56:33	2017-10-02 11:07:15	2017-10-04 19:55:00	2017-10-10 21:25:13	2017-10-18 00:00:00	7c396fd4830fd04220f754e42b4e5bff	3149	...	voucher	1	2.00	1	87285b34884572647811a353c7ac498a	3504c0cb71d7fa48d967e0e4c94d59d9	2017-10-06 11:07:15	29.99	8.72	utilidades_domesticas
2	e481f51cbdc54678b7cc49136f2d6af7	9ef432eb6251297304e76186b10a928d	delivered	2017-10-02 10:56:33	2017-10-02 11:07:15	2017-10-04 19:55:00	2017-10-10 21:25:13	2017-10-18 00:00:00	7c396fd4830fd04220f754e42b4e5bff	3149	...	voucher	1	18.59	1	87285b34884572647811a353c7ac498a	3504c0cb71d7fa48d967e0e4c94d59d9	2017-10-06 11:07:15	29.99	8.72	utilidades_domesticas
3	53cdb2fc8bc7dce0b6741e2150273451	b0830fb4747a6c6d20dea0b8c802d7ef	delivered	2018-07-24 20:41:37	2018-07-26 03:24:27	2018-07-26 14:31:00	2018-08-07 15:27:45	2018-08-13 00:00:00	af07308b275d755c9edb36a90c618231	47813	...	boleto	1	141.46	1	595fac2a385ac33a80bd5114aec74eb8	289cdb325fb7e7f891c38608bf9e0962	2018-07-30 03:24:27	118.70	22.76	perfumaria
4	47770eb9100c2d0c44946d9cf07ec65d	41ce2a54c0b03bf3443c3d931a367089	delivered	2018-08-08 08:38:49	2018-08-08 08:55:23	2018-08-08 13:50:00	2018-08-17 18:06:29	2018-09-04 00:00:00	3a653a41f6f9fc3d2a113cf8398680e8	75265	...	credit_card	3	179.12	1	aa4383b373c6aca5d8797843e5594415	4869f7a5dfa277a7dca6462dcf3b52b2	2018-08-13 08:55:23	159.90	19.22	automotivo
5 rows × 23 columns

```

4. Análise exploratória

```python

# Receita por estado

revenue_by_state = df_full.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False)
print(revenue_by_state)

# Visualização
revenue_by_state.plot(kind="bar", figsize=(12,6), title="Receita por Estado")

```

```python

# Ticket médio por cliente

ticket_medio = df_full.groupby("customer_id")["payment_value"].sum().mean()
print("Ticket médio por cliente:", ticket_medio)

Ticket médio por cliente: 205.82916647240668

# Frequência de compras por cliente

freq_compras = df_full.groupby("customer_id")["order_id"].nunique().mean()
print("Número médio de compras por cliente:", freq_compras)

Número médio de compras por cliente: 1.0

# 📊 Receita por Estado
# Quais são os estados que mais contribuem para o faturamento do e‑commerce?

top_states = revenue_by_state.nlargest(10)
sns.heatmap(top_states.to_frame(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Top 10 Estados por Receita")
plt.show()
```
<img width="521" height="435" alt="image" src="https://github.com/user-attachments/assets/98bfdc52-4a03-45f8-ab62-d2c94003d923" />


```python

# Relação entre valor do pagamento e quantidade de itens
# Relação entre o número de itens comprados em um pedido e o valor total pago
# Existe correlação entre quantidade de itens e valor pago?

df_full["num_items"] = df_full.groupby("order_id")["order_item_id"].transform("count")
scatter_data = df_full.groupby("order_id")[["payment_value","num_items"]].sum().reset_index()

fig = px.scatter(scatter_data, x="num_items", y="payment_value",
                 title="Relação entre número de itens e valor pago",
                 labels={"num_items":"Quantidade de Itens","payment_value":"Valor Pago"})
fig.show()
```

<img width="1476" height="419" alt="image" src="https://github.com/user-attachments/assets/c3c8e15f-336a-406b-9332-6286bac0214a" />

📘 Roteiro: Previsão de Valor Gasto / Ticket Médio

1. Preparação dos dados

Variáveis de interesse:
 - total_spent → soma dos pagamentos por cliente
 - avg_ticket → média dos pagamentos por cliente
 - Features possíveis: número de pedidos, estado do cliente, quantidade média de itens por pedido.

```python

# Total gasto por cliente
df_customer_spent = df_full.groupby("customer_id")["payment_value"].sum().reset_index()
df_customer_spent.rename(columns={"payment_value":"total_spent"}, inplace=True)

# Ticket médio por cliente
df_customer_avg = df_full.groupby("customer_id")["payment_value"].mean().reset_index()
df_customer_avg.rename(columns={"payment_value":"avg_ticket"}, inplace=True)

# Número de pedidos
df_customer_orders = df_full.groupby("customer_id")["order_id"].nunique().reset_index()
df_customer_orders.rename(columns={"order_id":"num_orders"}, inplace=True)

# Quantidade média de itens
df_customer_items = df_full.groupby("customer_id")["order_item_id"].count().reset_index()
df_customer_items.rename(columns={"order_item_id":"num_items"}, inplace=True)

# Juntar tudo
df_ml = df_customer_spent.merge(df_customer_avg, on="customer_id") \
                         .merge(df_customer_orders, on="customer_id") \
                         .merge(df_customer_items, on="customer_id") \
                         .merge(df_customers[["customer_id","customer_state"]], on="customer_id")

df_ml.head()

customer_id	total_spent	avg_ticket	num_orders	num_items	customer_state

0	00012a2ce6f8dcda20d059ce98491703	114.74	114.74	1	1	SP
1	000161a058600d5901f007fab4c27140	67.41	67.41	1	1	MG
2	0001fd6190edaaf884bcaf3d49edf079	195.42	195.42	1	1	ES
3	0002414f95344307404f0ace7a26f1d5	179.35	179.35	1	1	MG
4	000379cdec625522490c315e70c7a9fb	107.01	107.01	1	1	SP

```

Modelagem

ML: Random Forest Regressor

```python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Codificar estado
le = LabelEncoder()
df_ml["state_encoded"] = le.fit_transform(df_ml["customer_state"])

# Features e target
X = df_ml[["num_orders","num_items","state_encoded"]]
y = df_ml["total_spent"]   # ou "avg_ticket"

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2_score(y_test, y_pred))

MSE: 224013.45292086428
RMSE: 473.30059467622084
R²: 0.2312469286197535

```

📊 Interpretação

RMSE ≈ 473 → em média, o modelo erra cerca de R$473 na previsão do valor gasto por cliente.
R² ≈ 0.23 → o modelo explica apenas 23% da variação do gasto. 
Isso indica que os features usados (número de pedidos, itens e estado) não são suficientes para capturar bem o comportamento de gasto.
Em resumo: o Random Forest Regressor conseguiu capturar parte da variação, mas ainda há espaço para melhorar com novas variáveis e modelos mais sofisticados.

```python

# 🔎 Visualização da performance

# Gráfico de dispersão comparando valores reais vs previstos


fig = px.scatter(x=y_test, y=y_pred,
                 labels={"x":"Valor Real","y":"Valor Previsto"},
                 title="Valores Reais vs Previstos (Random Forest)")
fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
              x1=y_test.max(), y1=y_test.max(),
              line=dict(color="red", dash="dash"))
fig.show()

```
<img width="1472" height="416" alt="image" src="https://github.com/user-attachments/assets/5d6559ae-42eb-4f39-abc1-47c3ec4f0bab" /> <p><p><p>


📂 Adicionando novas Features

``` python

# 1. Tipo de pagamento
# Codificação do tipo de pagamento

df_payment_type = df_full.groupby("customer_id")["payment_type"].agg(lambda x: x.mode()[0]).reset_index()
df_payment_type.rename(columns={"payment_type":"main_payment_type"}, inplace=True)


# 2. Número de parcelas (payment_installments)
# Média de parcelas por cliente.

df_installments = df_full.groupby("customer_id")["payment_installments"].mean().reset_index()
df_installments.rename(columns={"payment_installments":"avg_installments"}, inplace=True)


# 3. Tempo médio de entrega
# Diferença entre data de entrega e data de compra.

df_full["delivery_time"] = (
    pd.to_datetime(df_full["order_delivered_customer_date"]) -
    pd.to_datetime(df_full["order_purchase_timestamp"])
).dt.days

df_delivery = df_full.groupby("customer_id")["delivery_time"].mean().reset_index()
df_delivery.rename(columns={"delivery_time":"avg_delivery_time"}, inplace=True)


# 4. Categoria dos produtos (product_category_name)
# Categoria mais frequente comprada por cliente.


df_category = df_full.groupby("customer_id")["product_category_name"] \
                     .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown") \
                     .reset_index()

df_category.rename(columns={"product_category_name":"main_category"}, inplace=True)

```



♻️ Unindo as novas Features

``` python

df_ml_expanded = df_ml.merge(df_payment_type, on="customer_id") \
                      .merge(df_installments, on="customer_id") \
                      .merge(df_delivery, on="customer_id") \
                      .merge(df_category, on="customer_id")

df_ml_expanded.head()


customer_id	total_spent	avg_ticket	num_orders	num_items	customer_state	state_encoded	main_payment_type	avg_installments	avg_delivery_time	main_category

0	00012a2ce6f8dcda20d059ce98491703	114.74	114.74	1	1	SP	25	credit_card	8.0	13.0	brinquedos
1	000161a058600d5901f007fab4c27140	67.41	67.41	1	1	MG	10	credit_card	5.0	9.0	beleza_saude
2	0001fd6190edaaf884bcaf3d49edf079	195.42	195.42	1	1	ES	7	credit_card	10.0	5.0	bebes
3	0002414f95344307404f0ace7a26f1d5	179.35	179.35	1	1	MG	10	boleto	1.0	28.0	cool_stuff
4	000379cdec625522490c315e70c7a9fb	107.01	107.01	1	1	SP	25	boleto	1.0	11.0	cama_mesa_banho


```
🔎 Preparação dos dados e treinamento do modelo Random Forest Regressor

``` python

"""
Este bloco de código prepara os dados, treina um modelo de Random Forest Regressor e avalia seu desempenho.

Passos:
1.  **Codificação de Features Categóricas**: Utiliza `LabelEncoder` para transformar as colunas categóricas `main_payment_type` e `main_category` em representações numéricas (`payment_encoded` e `category_encoded`).
2.  **Definição de Features (X) e Target (y)**: Define as variáveis independentes (features) para o modelo, que incluem `num_orders`, `num_items`, `state_encoded`, `avg_installments`, `avg_delivery_time`, `payment_encoded` e `category_encoded`. A variável dependente (target) é `total_spent`.
3.  **Divisão Treino/Teste**: Divide o dataset em conjuntos de treino (70%) e teste (30%) usando `train_test_split` para avaliar a generalização do modelo.
4.  **Treinamento do Modelo**: Inicializa e treina um `RandomForestRegressor` com `n_estimators=200` e `random_state=42` no conjunto de treino.
5.  **Avaliação do Modelo**: Realiza previsões no conjunto de teste (`y_pred`) e calcula métricas de desempenho:
    *   **MSE (Mean Squared Error)**: Média dos quadrados dos erros.
    *   **RMSE (Root Mean Squared Error)**: Raiz quadrada do MSE, indicando o desvio padrão dos resíduos.
    *   **R² (Coefficient of Determination)**: Proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Um R² mais alto indica um melhor ajuste do modelo.

Resultados esperados:
Os valores de MSE, RMSE e R² são impressos para avaliar o quão bem o modelo consegue prever o valor total gasto pelos clientes.
"""

# Codificação
le_payment = LabelEncoder()
df_ml_expanded["payment_encoded"] = le_payment.fit_transform(df_ml_expanded["main_payment_type"])

le_category = LabelEncoder()
df_ml_expanded["category_encoded"] = le_category.fit_transform(df_ml_expanded["main_category"])

# Features e target
X = df_ml_expanded[["num_orders","num_items","state_encoded",
                    "avg_installments","avg_delivery_time",
                    "payment_encoded","category_encoded"]]
y = df_ml_expanded["total_spent"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2_score(y_test, y_pred))

MSE: 172678.14218104785
RMSE: 415.5455957907
R²: 0.40741571351605377

```

📊 Interpretação

A inclusão de tipo de pagamento, número de parcelas, tempo médio de entrega e categoria principal trouxe mais poder explicativo.

Isso mostra que o comportamento de gasto dos clientes não depende apenas de número de pedidos/itens, mas também de como pagam, 
em quantas vezes, quanto tempo esperam pela entrega e o tipo de produto comprado.

O modelo ainda pode melhorar, mas já está em um modelo para análise de negócio.



_________________________________________________________________________________________________________________________________________

📚 Resumo Geral


🗳️ Esta é uma análise abrangente dos dados da Olist. Primeiramente, os dados foram carregados, explorados e limpos. Em seguida, foi feita uma análise exploratória para entender a distribuição de clientes por estado e cidade, receita por estado, ticket médio e frequência de compras.

Depois, foi desenvolvido um modelo de Machine Learning (RandomForestRegressor) para prever o valor gasto pelos clientes. A primeira versão do modelo, usando número de pedidos, itens e estado do cliente, obteve um R² de aproximadamente 0.23. Ao expandir as features com tipo de pagamento, número de parcelas, tempo médio de entrega e categoria principal do produto, o modelo melhorou significativamente, alcançando um R² de aproximadamente 0.41. Essa melhoria destaca a importância das novas variáveis para entender o comportamento de gastos dos clientes.
