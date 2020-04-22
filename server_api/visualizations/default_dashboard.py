import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Select, TextInput, Button, Text, Div
from shapely.geometry import Point
import shap

from coord2vec.common.itertools import flatten
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.evaluation.visualizations.bokeh_plots import bokeh_multiple_pr_curves, log_current_map_position_js, \
    bokeh_score_histogram, feature_importance, bokeh_scored_polygons_folium
from coord2vec.experiments.experiment_loader import load_experiment_results, load_separate_model_results
from coord2vec.common.geographic.geo_utils import get_closest_geo
from coord2vec.feature_extraction.osm.osm_utils import extract_buildings_from_polygons

color_gradients = [
    # green to red
    LinearSegmentedColormap.from_list('rgb', [(0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0)], N=256, gamma=1.0),
    # deep blue to light blue
    LinearSegmentedColormap.from_list('rgb', [(0.20442906574394465, 0.29301038062283735, 0.35649365628604385, 0),
                                              (0.20898116109188775, 0.38860438292964244, 0.5173343585800334, 0),
                                              (0.21341022683583238, 0.4816147635524798, 0.6738280148660771, 0),
                                              (0.2818813276944765, 0.5707599641163655, 0.7754914776368064, 0),
                                              (0.41069332308086637, 0.6512213251313598, 0.8168294245802896, 0),
                                              (0.5430834294502115, 0.733917723952326, 0.8593156478277586, 0)],
                                      N=256, gamma=1.0)]


class DefaultDashboard:
    start_location = None
    folium_zoom = 14

    def __init__(self, task: TaskHandler, results_dir=BUILDING_RESULTS_DIR):
        self.task = task
        self.main_panel = column()
        self.folium_name = task.__str__()

        # unpack building experiment results
        self.kfold = 0
        self.model_idx = 0
        # self.full_results = load_experiment_results(results_dir)['model_results']
        self.full_results = load_separate_model_results(results_dir)['model_results']
        self.model_results = self.full_results[self.model_idx]
        self.geos_train, self.X_train_df, self.y_train, self.train_probas, \
        self.geos_test, self.X_test_df, self.y_test, self.test_probas, \
        self.models, self.model_names, self.auc_scores = self._extract_model_results(self.full_results, self.model_idx,
                                                                                     self.kfold)

        self.num_kfold = len(self.full_results[0]['train_idx'])
        self.features_df = pd.concat([self.X_train_df, self.X_test_df])
        self.features_kmeans = shap.kmeans(self.features_df, 10)
        self.all_probas = np.concatenate([self.train_probas[self.model_idx], self.test_probas[self.model_idx]])
        self.all_probas_df = pd.Series(data=self.all_probas, index=self.features_df.index)

        self.mean_auc = self.mean_auc_panel(self.model_names, self.auc_scores)
        plot = self.bokeh_plot()
        self.main_panel.children.append(plot)

    @staticmethod
    def mean_auc_panel(model_names, auc_scores):
        text = "<h3>Mean AUC results </h3>"
        for i, model_name in enumerate(model_names):
            model_mean = np.mean(auc_scores[i])
            text += f"<br> <b>{model_name}:</b> \t {round(model_mean, 3)}"
        return Div(text=text)

    def bokeh_plot(self):
        # create precision recall curve
        test_probas_df = pd.DataFrame(
            {model_name: test_proba for model_name, test_proba in zip(self.model_names, self.test_probas)})
        pr_curve = bokeh_multiple_pr_curves(test_probas_df, self.y_test.values)

        # create feature histogram
        self.feature_select = Select(title="Choose Feature: ", value=self.X_train_df.columns[0],
                                     options=list(self.X_train_df.columns), width=200)
        self.feature_select.on_change('value', self.choose_feature_callback)
        # self.feature_select.js_on_change('value', log_current_map_position_js(self.folium_name))
        hist = bokeh_score_histogram(self.X_train_df.iloc[:, 0], self.y_train > 0.5)

        # create model select
        self.model_select = Select(title="Choose Model: ", value=self.model_names[0],
                                   options=list(self.model_names), width=300)
        self.kfold_select = Select(title="Choose Kfold: ", value='0',
                                   options=list(map(str, range(self.num_kfold))), width=150)
        self.run_button = Button(label="RUN", button_type="success", width=100)
        self.run_button.js_on_click(log_current_map_position_js(self.folium_name))
        self.run_button.on_click(self.run_callback)

        # build input for map click coords
        lon_text = TextInput(value="0", title='lon:', width=100)
        lat_text = TextInput(value="0", title='lat:', width=100)
        self.lonlat_text_inputs = [lon_text, lat_text]
        feature_importance_button = Button(label='Calculate Feature Values')
        # feature_importance_button.js_on_click(log_current_map_position_js(self.folium_name))
        feature_importance_button.on_click(self.sample_feature_importance_update)
        importance_fig, _ = self.build_importance_figure(Point(0, 0))

        # create folium figure
        folium_fig = bokeh_scored_polygons_folium([self.all_probas_df],
                                                  [True], train_geos=self.geos_train,
                                                  test_geos=self.geos_test, start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700, lonlat_text_inputs=self.lonlat_text_inputs)

        # put it all together
        self.left_column = column(pr_curve, self.feature_select, hist)
        self.folium_column = column(row(self.model_select, self.kfold_select, self.run_button), folium_fig,
                                    self.mean_auc)
        self.importance_and_val_column = column(row(lon_text, lat_text), feature_importance_button, importance_fig)

        return row(self.left_column, self.folium_column, self.importance_and_val_column)

    def sample_feature_importance_update(self):
        lat, lon = self.get_click_lonlat()
        point = Point([lon, lat])
        importance_figure, closest_geo = self.build_importance_figure(point)

        self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        self.importance_and_val_column.children[-1] = importance_figure

    def choose_feature_callback(self, attr, old, new):
        print(f"Chose feature: {new}")
        new_hist = bokeh_score_histogram(self.X_train_df[new], self.y_train > 0.5)
        self.left_column.children[-1] = new_hist

    def build_importance_figure(self, point: Point):
        if point.x == 0 and point.y == 0:
            closest_geo = point
            closest_geo_df = pd.DataFrame([self.features_df.mean().values], columns=self.features_df.columns,
                                          index=['aggregated'])
        else:
            closest_geo = get_closest_geo(point, self.features_df.index)
            closest_geo_df = self.features_df[self.features_df.index == closest_geo]

        model = self.models[self.model_names.index(self.model_select.value)]
        explainer = shap.KernelExplainer(model.predict, self.features_kmeans)
        shap_values_model = explainer.shap_values(closest_geo_df, silent=True)[0]  # TODO: very slow
        # importance_fig = shap.force_plot(explainer.expected_value, shap_values_model, closest_geo_df)

        sort_indices = np.argsort(np.abs(shap_values_model))
        importance_fig = feature_importance(list(self.X_train_df.columns[sort_indices]),
                                            shap_values_model[sort_indices])

        # shap.save_html('importance_fig.html', importance_fig)
        # importance_fig = Div(text="""<iframe>
        #                                 src="importance_fig.html"
        #                             </iframe>""")

        return importance_fig, closest_geo

    def run_callback(self):
        self.model_idx = self.model_names.index(self.model_select.value)
        self.kfold = int(self.kfold_select.value)


        # set all the parameters of the new kfold
        self.geos_train, self.X_train_df, self.y_train, self.train_probas, \
        self.geos_test, self.X_test_df, self.y_test, self.test_probas, \
        self.models, self.model_names, self.auc_scores = self._extract_model_results(self.full_results, self.model_idx,
                                                                                     self.kfold)

        self.num_kfold = len(self.full_results[0]['train_idx'])
        self.features_df = pd.concat([self.X_train_df, self.X_test_df])
        self.features_kmeans = shap.kmeans(self.features_df, 10)
        self.all_probas = np.concatenate([self.train_probas[self.model_idx], self.test_probas[self.model_idx]])
        self.all_probas_df = pd.Series(data=self.all_probas, index=self.features_df.index)

        # create new precision recall curve
        test_probas_df = pd.DataFrame(
            {model_name: test_proba for model_name, test_proba in zip(self.model_names, self.test_probas)})
        pr_curve = bokeh_multiple_pr_curves(test_probas_df, self.y_test.values)
        self.left_column.children[0] = pr_curve

        # create new folium map
        print(f"Choose model: {self.model_select.value}\t kfold: {self.kfold}")
        new_folium = bokeh_scored_polygons_folium([self.all_probas_df],
                                                  [True], train_geos=self.geos_train,
                                                  test_geos=self.geos_test, start_zoom=self.folium_zoom,
                                                  start_location=self.start_location, file_name=self.folium_name,
                                                  width=700, lonlat_text_inputs=self.lonlat_text_inputs)
        self.folium_column.children[-2] = new_folium

        self.sample_feature_importance_update()

    def get_click_lonlat(self):
        lon = self.lonlat_text_inputs[0].value
        lat = self.lonlat_text_inputs[1].value
        lon = float(lon) if lon != "" else 0
        lat = float(lat) if lat != "" else 0
        return lat, lon

    def _extract_model_results(self, full_results, model_idx, kfold):
        model_results = full_results[model_idx]
        train_idx, test_idx = model_results['train_idx'][kfold], model_results['test_idx'][kfold]

        # X and y
        X_train_df, X_test_df = model_results['X_df'].iloc[train_idx], model_results['X_df'].iloc[test_idx]
        y_train, y_test = model_results['y'][train_idx], model_results['y'][test_idx]

        # geos
        geos_train_idx, geos_test_idx = model_results['geos_kfold_split'][kfold]
        geos_train, geos_test = model_results['geos'][geos_train_idx], model_results['geos'][geos_test_idx]

        # probas
        train_probas = [results['probas'][kfold][train_idx] for results in full_results]
        test_probas = [results['probas'][kfold][test_idx] for results in full_results]

        # models
        model_names = [results['model_name'] for results in full_results]
        models = [results['models'][kfold] for results in full_results]
        scores = [results['auc_scores'] for results in full_results]

        return geos_train, X_train_df, y_train, train_probas, \
               geos_test, X_test_df, y_test, test_probas, \
               models, model_names, scores
