"""
Módulo de inferência para previsão de defasagem escolar.

Este módulo fornece uma interface simples para carregar o modelo
e realizar predições, facilitando a integração com APIs (FastAPI).

Exemplo de uso na API:
    from src.model.predict import ModelPredictor
    
    predictor = ModelPredictor()
    resultado = predictor.predict(dados_aluno)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import joblib
import pandas as pd
import numpy as np

from .feature_engineering import build_features


# Diretório padrão do modelo
MODEL_DIR = Path(__file__).parent.parent.parent / "app" / "model"

# Classes possíveis
CLASS_ORDER = ["baixo", "medio", "alto"]


class ModelPredictor:
    """
    Classe para realizar predições de defasagem escolar.

    Esta classe encapsula o carregamento do modelo e a lógica de predição,
    fornecendo uma interface simples para uso em APIs.

    Attributes:
        pipeline: Pipeline do sklearn com preprocessador e modelo
        metadata: Metadados do modelo (métricas, configurações)
        is_loaded: Indica se o modelo foi carregado

    Example:
        >>> predictor = ModelPredictor()
        >>> predictor.load()
        >>> resultado = predictor.predict({"idade": 15, "nota": 7.5, ...})
        >>> print(resultado["classe"])
        'baixo'
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa o preditor.

        Args:
            model_path: Caminho para o arquivo do modelo.
                       Se None, usa o caminho padrão em app/model/
        """
        self.model_path = Path(
            model_path) if model_path else MODEL_DIR / "model.joblib"
        self.metadata_path = self.model_path.parent / "model_metadata.joblib"
        self.pipeline = None
        self.metadata = None
        self.is_loaded = False

    def load(self) -> "ModelPredictor":
        """
        Carrega o modelo do disco.

        Returns:
            Self para permitir encadeamento

        Raises:
            FileNotFoundError: Se o arquivo do modelo não existir
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {self.model_path}. "
                "Execute o treinamento primeiro."
            )

        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            self.metadata = joblib.load(self.metadata_path)
        else:
            self.metadata = {}

        self.is_loaded = True
        return self

    def _ensure_loaded(self) -> None:
        """Garante que o modelo está carregado."""
        if not self.is_loaded:
            self.load()

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """
        Realiza predição para um ou mais registros.

        Args:
            data: Dados para predição. Pode ser:
                  - Dict com dados de um único aluno
                  - DataFrame com múltiplos alunos
            return_probabilities: Se True, retorna probabilidades de cada classe

        Returns:
            Dict com resultado da predição:
            {
                "classe": str ou List[str],  # Classe predita
                "probabilidades": Dict (opcional),  # Probabilidades por classe
                "sucesso": bool,  # Indica se a predição foi bem-sucedida
                "erro": str (opcional)  # Mensagem de erro se houver
            }

        Example:
            >>> resultado = predictor.predict({
            ...     "Idade_Aluno_2024": 15,
            ...     "PEDRA_2021": 50,
            ...     ...
            ... })
            >>> print(resultado)
            {'classe': 'baixo', 'sucesso': True}
        """
        self._ensure_loaded()

        try:
            # Converte dict para DataFrame se necessário
            if isinstance(data, dict):
                df = pd.DataFrame([data])
                single_record = True
            else:
                df = data.copy()
                single_record = len(df) == 1

            # Aplica feature engineering (sem target)
            X = build_features(df, include_target=False)

            # Realiza predição
            predictions = self.pipeline.predict(X)

            # Prepara resultado
            result: Dict[str, Any] = {
                "sucesso": True,
            }

            if single_record:
                result["classe"] = predictions[0]
            else:
                result["classe"] = predictions.tolist()

            # Adiciona probabilidades se solicitado
            if return_probabilities:
                probas = self.pipeline.predict_proba(X)
                if single_record:
                    result["probabilidades"] = {
                        classe: float(probas[0][i])
                        for i, classe in enumerate(CLASS_ORDER)
                    }
                else:
                    result["probabilidades"] = [
                        {
                            classe: float(probas[j][i])
                            for i, classe in enumerate(CLASS_ORDER)
                        }
                        for j in range(len(probas))
                    ]

            return result

        except Exception as e:
            return {
                "sucesso": False,
                "erro": str(e),
            }

    def predict_batch(
        self,
        data: pd.DataFrame,
        return_probabilities: bool = False,
    ) -> pd.DataFrame:
        """
        Realiza predição em lote, retornando DataFrame com resultados.

        Args:
            data: DataFrame com dados dos alunos
            return_probabilities: Se True, adiciona colunas de probabilidade

        Returns:
            DataFrame original com colunas adicionais:
            - 'predicao': Classe predita
            - 'prob_baixo', 'prob_medio', 'prob_alto' (se return_probabilities=True)
        """
        self._ensure_loaded()

        # Aplica feature engineering
        X = build_features(data, include_target=False)

        # Realiza predições
        predictions = self.pipeline.predict(X)

        # Cria DataFrame de resultado
        result_df = data.copy()
        result_df["predicao"] = predictions

        if return_probabilities:
            probas = self.pipeline.predict_proba(X)
            for i, classe in enumerate(CLASS_ORDER):
                result_df[f"prob_{classe}"] = probas[:, i]

        return result_df

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado.

        Returns:
            Dict com informações do modelo:
            - model_name: Nome do modelo
            - classes: Classes possíveis
            - metrics: Métricas de avaliação
        """
        self._ensure_loaded()

        info = {
            "model_name": self.metadata.get("model_name", "LogisticRegression"),
            "classes": CLASS_ORDER,
            "descricao_classes": {
                "baixo": "Defasagem baixa ou nenhuma (Defas >= 0)",
                "medio": "Defasagem média (Defas entre -1 e -2)",
                "alto": "Defasagem alta (Defas <= -3)",
            },
        }

        if self.metadata.get("metrics"):
            info["metrics"] = self.metadata["metrics"]

        return info

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica se o modelo está funcionando corretamente.

        Útil para endpoints de health check na API.

        Returns:
            Dict com status do modelo
        """
        try:
            self._ensure_loaded()
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_path": str(self.model_path),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
            }


# Instância singleton para uso direto
_predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """
    Retorna uma instância singleton do ModelPredictor.

    Útil para evitar recarregar o modelo a cada requisição na API.

    Returns:
        Instância do ModelPredictor já carregada

    Example:
        from src.model.predict import get_predictor

        predictor = get_predictor()
        resultado = predictor.predict(dados)
    """
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
        _predictor.load()
    return _predictor


def predict(
    data: Union[Dict[str, Any], pd.DataFrame],
    return_probabilities: bool = False,
) -> Dict[str, Any]:
    """
    Função de conveniência para predição rápida.

    Usa a instância singleton do ModelPredictor.

    Args:
        data: Dados para predição (dict ou DataFrame)
        return_probabilities: Se True, retorna probabilidades

    Returns:
        Dict com resultado da predição

    Example:
        from src.model.predict import predict

        resultado = predict({"idade": 15, ...})
        print(resultado["classe"])  # 'baixo'
    """
    return get_predictor().predict(data, return_probabilities)
