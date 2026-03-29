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

4.Análise exploratória

```python

# Receita por estado

revenue_by_state = df_full.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False)
print(revenue_by_state)

# Visualização
revenue_by_state.plot(kind="bar", figsize=(12,6), title="Receita por Estado")

```

