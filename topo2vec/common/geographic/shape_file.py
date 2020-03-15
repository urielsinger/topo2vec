import fiona
import pandas as pd


class ShapeFile:
    """
    This class will handle shape files. reading, parsing and visualizing.
    """

    def __init__(self, shp_path):
        self.shp_path = shp_path
        self.records = fiona.open(self.shp_path,  encoding = 'utf-8')

    def to_dataframe(self) -> pd.DataFrame:
        '''
        Returns: a dataframe representing the shapefile
        '''

        fields = [field[0] for field in self.sf.fields][1:]
        shps = [s.points for s in self.sf.shapes()]

        df = pd.DataFrame(recoreds, columns=fields)
        df = df.assign(coords=shps)

        return df
