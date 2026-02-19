import joblib

# Carrega o pipeline
pipeline = joblib.load('app/model/model.joblib')

print('ESTRUTURA DO PIPELINE:')
print(f'Steps: {[step[0] for step in pipeline.steps]}')
print()

# Acessa o preprocessor
preprocessor = pipeline.named_steps['preprocessor']
print('PREPROCESSOR (ColumnTransformer):')
print(f'Transformers: {[t[0] for t in preprocessor.transformers_]}')
print()

# Acessa o StandardScaler
numeric_pipeline = preprocessor.named_transformers_['numeric']
scaler = numeric_pipeline.named_steps['scaler']

print('STANDARD SCALER (NORMALIZACAO):')
print(f'Medias salvas (primeiras 5): {scaler.mean_[:5]}')
print(f'Escalas salvas (primeiras 5): {scaler.scale_[:5]}')
print(f'Total features numericas: {len(scaler.mean_)}')
print()

# Acessa o OneHotEncoder
categorical_pipeline = preprocessor.named_transformers_['categorical']
encoder = categorical_pipeline.named_steps['onehot']
print('ONEHOT ENCODER (CATEGORICAS):')
print(f'Categorias salvas: {encoder.categories_}')
print()

# Classificador
classifier = pipeline.named_steps['classifier']
print('LOGISTIC REGRESSION:')
print(f'Classes: {classifier.classes_}')
print(f'Coeficientes shape: {classifier.coef_.shape}')
