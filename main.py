from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd

df: pd.DataFrame = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")

# 1. 加载预训练模型和 tokenizer
model_name = 'bert-base-uncased'  # 或其他预训练模型名称
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6) # 2 分类

# 2. 数据预处理
sentences = df.drop(columns=['label'], axis=1).values.flatten().tolist()
labels = df['label'].tolist()

print("Start Tokenizing")
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='tf')
print("Tokenizing Completed")

# 3. 创建 TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((dict(encoded_input), labels))
dataset = dataset.batch(2) # 批次大小

# 4. 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 5. 训练模型
model.fit(dataset, epochs=3)

# 6. 推理

new_sentences = ["I'm good"]
new_encoded_input = tokenizer(new_sentences, padding=True, truncation=True, return_tensors='tf')
predictions = model(new_encoded_input)
print(predictions.logits)
probabilities = tf.nn.softmax(predictions.logits)
print(probabilities) # 输出 logits