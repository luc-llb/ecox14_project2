"""
Utilitário para processamento de gêneros de anime em one-hot encoding.

Este módulo fornece funções para transformar a coluna 'genres' do dataset de animes
em uma matriz de one-hot encoding, facilitando análises de machine learning.
"""

import pandas as pd
import os
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Gêneros a serem filtrados das análises
GENEROS_FILTRADOS = ['Ecchi', 'Boys Love', 'Girls Love', 'Erotica', 'Hentai']

def processar_generos_onehot(
        df_animes: pd.DataFrame, 
        salvar_arquivo: bool = True, 
        caminho_arquivo: str = './datas/generos_onehot.csv'
    ) -> pd.DataFrame:
    """
    Processa a coluna de gêneros do dataset de animes e cria uma matriz one-hot encoding.
    """
    print("Iniciando processamento dos gêneros...")
    
    # Remover animes sem gênero
    df_clean = df_animes[df_animes['genres'].notna()].copy()
    print(f"Animes com gêneros válidos: {len(df_clean)}")
    
    # Processar string de gêneros para lista E FILTRAR gêneros adultos
    generos_processados = []
    for idx, genres_str in enumerate(df_clean['genres']):
        try:
            # Remover colchetes externos e processar como lista
            if isinstance(genres_str, str):
                genres_list = ast.literal_eval(genres_str)
                if isinstance(genres_list, list):
                    # FILTRAR gêneros adultos ANTES do MultiLabelBinarizer
                    genres_filtrados = [g for g in genres_list if g not in GENEROS_FILTRADOS]
                    generos_processados.append(genres_filtrados)
                else:
                    generos_processados.append([])
            else:
                generos_processados.append([])
        except (ValueError, SyntaxError):
            # Se falhar, tentar processamento manual
            if isinstance(genres_str, str):
                clean_str = genres_str.strip("[]'\"")
                genres_list = [g.strip().strip("'\"") for g in clean_str.split(',') if g.strip()]
                # FILTRAR gêneros adultos ANTES do MultiLabelBinarizer
                genres_filtrados = [g for g in genres_list if g not in GENEROS_FILTRADOS]
                generos_processados.append(genres_filtrados)
            else:
                generos_processados.append([])
    
    print(f"Gêneros processados (após filtro): {len(generos_processados)}")
    print(f"Gêneros adultos removidos na fonte: {GENEROS_FILTRADOS}")
    
    # Criar one-hot encoding usando MultiLabelBinarizer (já com dados filtrados)
    mlb = MultiLabelBinarizer()
    generos_onehot = mlb.fit_transform(generos_processados)
    df_onehot = pd.DataFrame(generos_onehot, columns=mlb.classes_, index=df_clean.index)
    
    # Adicionar colunas de identificação
    df_resultado = pd.concat([
        df_clean[['animeID', 'title', 'year']].reset_index(drop=True),
        df_onehot.reset_index(drop=True)
    ], axis=1)
    
    print(f"Matriz one-hot criada: {df_resultado.shape}")
    print(f"Gêneros únicos encontrados: {len(mlb.classes_)}")
    print(f"Primeiros gêneros: {list(mlb.classes_[:10])}")
    
    # Salvar em arquivo
    if salvar_arquivo:
        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
        df_resultado.to_csv(caminho_arquivo, index=False)
        print(f"Arquivo salvo em: {caminho_arquivo}")
    
    return df_resultado


def carregar_ou_processar_generos(
        caminho_dataset: str ='./datas/animes.csv', 
        caminho_onehot: str ='./datas/generos_onehot.csv'
    ) -> pd.DataFrame:
    
    # Se o arquivo existir realiza a leitura, do contrario faz o processamento 
    if os.path.exists(caminho_onehot):
        print(f"Carregando arquivo processado: {caminho_onehot}")
        df_generos = pd.read_csv(caminho_onehot)
        print(f"Dados carregados: {df_generos.shape}")
        # Verificar se ainda há gêneros adultos (caso o arquivo seja antigo)
        df_generos = filtrar_generos_adultos(df_generos)
        return df_generos
    else:
        print(f"Arquivo processado não encontrado. Processando {caminho_dataset}...")
        df_animes = pd.read_csv(caminho_dataset)
        # Agora o processamento já filtra na fonte, mas verificamos por segurança
        df_generos = processar_generos_onehot(df_animes, salvar_arquivo=True, 
                                                caminho_arquivo=caminho_onehot)
        return df_generos


def filtrar_generos_adultos(df_generos: pd.DataFrame) -> pd.DataFrame:
    """
    Remove gêneros adultos/sensíveis das análises.
    """
    # Identificar colunas de gêneros a serem removidas
    colunas_para_remover = [col for col in df_generos.columns if col in GENEROS_FILTRADOS]
    
    if colunas_para_remover:
        print(f"Removendo gêneros adultos: {colunas_para_remover}")
        df_filtrado = df_generos.drop(columns=colunas_para_remover)
        print(f"Shape antes: {df_generos.shape}, Shape após filtragem: {df_filtrado.shape}")
        return df_filtrado
    else:
        print("Nenhum gênero adulto encontrado para remover.")
        return df_generos


def obter_colunas_generos_limpos(df_generos: pd.DataFrame) -> list:
    """
    Retorna lista das colunas de gêneros após filtrar gêneros adultos.
    """
    df_filtrado = filtrar_generos_adultos(df_generos)
    return [col for col in df_filtrado.columns 
            if col not in ['animeID', 'title', 'year']]


def obter_generos_por_ano(df_generos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega gêneros por ano para análise temporal.
    """
    # Verificar se ainda há gêneros adultos (por segurança, caso arquivo seja antigo)
    df_generos = filtrar_generos_adultos(df_generos)
    
    # Remove demais colunas
    colunas_generos = [col for col in df_generos.columns 
                        if col not in ['animeID', 'title', 'year']]
    
    # Filtrar anos de 1970 a 2024
    df_clean = df_generos[df_generos['year'].notna()].copy()
    df_clean = df_clean[(df_clean['year'] >= 1970) & (df_clean['year'] <= 2024)]

    generos_por_ano = df_clean.groupby('year')[colunas_generos].mean().reset_index()
    print(f"Dados agregados por ano: {generos_por_ano.shape}")
    print(f"Anos cobertos: {generos_por_ano['year'].min()} - {generos_por_ano['year'].max()}")
    
    return generos_por_ano


if __name__ == "__main__":
    print("Testando processamento de gêneros...")
    df_resultado = carregar_ou_processar_generos()
    print("\nPrimeiras linhas do resultado:")
    print(df_resultado.head())
    
    print("\nTestando agregação por ano:")
    df_por_ano = obter_generos_por_ano(df_resultado)
    print(df_por_ano.head())