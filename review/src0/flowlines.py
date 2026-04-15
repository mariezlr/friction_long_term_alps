"""
Module pour la gestion des flowlines (lignes d'écoulement) glaciaires.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, List, Optional


class FlowlineManager:
    """
    Classe pour gérer les flowlines des différents glaciers.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialise le gestionnaire de flowlines.
        
        Parameters
        ----------
        data_dir : Path
            Répertoire contenant les données
        """
        self.data_dir = Path(data_dir)
        self.flowlines = {}
        self.stake_indices = {}
    
    def compute_flowline(self,
                        flowline_points: List[List[float]],
                        num_points: int = 100,
                        use_x_as_param: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule une flowline interpolée à partir de points de contrôle.
        
        Parameters
        ----------
        flowline_points : list of list
            Points de contrôle [[x1, y1], [x2, y2], ...]
        num_points : int, optional
            Nombre de points dans la flowline interpolée
        use_x_as_param : bool, optional
            Si True, utilise x comme paramètre d'interpolation, sinon y
            
        Returns
        -------
        x_flowline : np.ndarray
            Coordonnées x de la flowline
        y_flowline : np.ndarray
            Coordonnées y de la flowline
        """
        x_points = [pt[0] for pt in flowline_points]
        y_points = [pt[1] for pt in flowline_points]
        
        if use_x_as_param:
            cs = CubicSpline(x_points, y_points, bc_type='natural')
            x_flowline = np.linspace(min(x_points), max(x_points), num_points)
            y_flowline = cs(x_flowline)
        else:
            cs = CubicSpline(y_points, x_points, bc_type='natural')
            y_flowline = np.linspace(min(y_points), max(y_points), num_points)
            x_flowline = cs(y_flowline)
        
        return x_flowline, y_flowline
    
    def find_stake_index(self,
                        x_flowline: np.ndarray,
                        y_flowline: np.ndarray,
                        x_stake: float,
                        y_stake: float) -> int:
        """
        Trouve l'index du point de la flowline le plus proche d'un stake.
        
        Parameters
        ----------
        x_flowline : np.ndarray
            Coordonnées x de la flowline
        y_flowline : np.ndarray
            Coordonnées y de la flowline
        x_stake : float
            Coordonnée x du stake
        y_stake : float
            Coordonnée y du stake
            
        Returns
        -------
        int
            Index du point le plus proche sur la flowline
        """
        tree = cKDTree(np.column_stack((x_flowline, y_flowline)))
        _, idx = tree.query([x_stake, y_stake])
        return idx
    
    def load_glacier_coords(self, glacier_name: str) -> pd.DataFrame:
        """
        Charge les coordonnées des stakes pour un glacier.
        
        Parameters
        ----------
        glacier_name : str
            Nom du glacier
            
        Returns
        -------
        pd.DataFrame
            DataFrame contenant les coordonnées
        """
        coord_file = self.data_dir / f"coord_stakes_{glacier_name.lower()}.csv"
        if coord_file.exists():
            return pd.read_csv(coord_file)
        else:
            raise FileNotFoundError(f"Fichier de coordonnées non trouvé: {coord_file}")
    
    def save_flowline(self,
                     glacier_name: str,
                     x_flowline: np.ndarray,
                     y_flowline: np.ndarray,
                     output_dir: Optional[Path] = None) -> Path:
        """
        Sauvegarde une flowline dans un fichier CSV.
        
        Parameters
        ----------
        glacier_name : str
            Nom du glacier
        x_flowline : np.ndarray
            Coordonnées x de la flowline
        y_flowline : np.ndarray
            Coordonnées y de la flowline
        output_dir : Path, optional
            Répertoire de sortie. Si None, utilise data_dir
            
        Returns
        -------
        Path
            Chemin du fichier créé
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({"x": x_flowline, "y": y_flowline})
        output_file = output_dir / f"{glacier_name}_flowline.csv"
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def save_stake_indices(self,
                          stake_data: List[Tuple[str, str, int]],
                          output_dir: Optional[Path] = None) -> Path:
        """
        Sauvegarde les indices des stakes le long des flowlines.
        
        Parameters
        ----------
        stake_data : list of tuple
            Liste de tuples (glacier, stake, idx)
        output_dir : Path, optional
            Répertoire de sortie. Si None, utilise data_dir
            
        Returns
        -------
        Path
            Chemin du fichier créé
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(stake_data, columns=["glacier", "stake", "idx"])
        output_file = output_dir / "flowline_idx.csv"
        df.to_csv(output_file, index=False)
        
        return output_file
    
    @staticmethod
    def sort_flowline_points(points: List[List[float]], 
                           sort_by: str = 'x') -> List[List[float]]:
        """
        Trie les points d'une flowline par x ou y.
        
        Parameters
        ----------
        points : list of list
            Points à trier [[x1, y1], [x2, y2], ...]
        sort_by : str, optional
            'x' ou 'y' pour le critère de tri
            
        Returns
        -------
        list of list
            Points triés
        """
        if sort_by == 'x':
            return sorted(points, key=lambda p: p[0])
        elif sort_by == 'y':
            return sorted(points, key=lambda p: p[1])
        else:
            raise ValueError("sort_by doit être 'x' ou 'y'")


def create_all_flowlines(data_dir: Path,
                        coords_file: Path,
                        output_dir: Optional[Path] = None) -> dict:
    """
    Crée toutes les flowlines pour les glaciers alpins du projet.
    
    Cette fonction est un wrapper pour recréer toutes les flowlines
    avec les configurations spécifiques du projet.
    
    Parameters
    ----------
    data_dir : Path
        Répertoire contenant les données brutes
    coords_file : Path
        Fichier CSV contenant les coordonnées des stakes
    output_dir : Path, optional
        Répertoire de sortie pour les flowlines
        
    Returns
    -------
    dict
        Dictionnaire contenant toutes les flowlines et indices
    """
    manager = FlowlineManager(data_dir)
    
    if output_dir is None:
        output_dir = data_dir / "flowlines"
    
    # Charger les coordonnées des stakes
    coords = pd.read_csv(coords_file)
    
    # Dictionnaire pour stocker les résultats
    results = {
        'flowlines': {},
        'indices': []
    }
    
    # Configuration des glaciers (à adapter selon vos besoins)
    glacier_configs = {
        'All': {
            'points': [[2635500, 1096500], [2636500, 1098500], 
                      [2639500, 1100000]],
            'stakes': ['101'],
            'use_x': True
        },
        'Arg': {
            'points': [[957500, 119000], [958652, 118002]],
            'stakes': ['wheel', '4', '5'],
            'use_x': True
        },
        # Ajouter d'autres glaciers ici...
    }
    
    # Créer les flowlines
    for glacier_name, config in glacier_configs.items():
        x_flow, y_flow = manager.compute_flowline(
            config['points'],
            use_x_as_param=config['use_x']
        )
        
        results['flowlines'][glacier_name] = (x_flow, y_flow)
        
        # Sauvegarder
        manager.save_flowline(glacier_name, x_flow, y_flow, output_dir)
        
        # Calculer les indices des stakes
        for stake in config['stakes']:
            row = coords[
                (coords['glacier'] == glacier_name) & 
                (coords['stake'] == stake)
            ]
            if not row.empty:
                x_stake = row['x'].values[0]
                y_stake = row['y'].values[0]
                idx = manager.find_stake_index(x_flow, y_flow, x_stake, y_stake)
                results['indices'].append((glacier_name, stake, idx))
    
    # Sauvegarder les indices
    manager.save_stake_indices(results['indices'], output_dir)
    
    return results
