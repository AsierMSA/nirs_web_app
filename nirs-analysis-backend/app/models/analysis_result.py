class AnalysisResult:
    """
    Data model for storing analysis results of NIRS data.

    Attributes:
    -----------
    accuracy : float
        The accuracy of the classification model.
    best_classifier : str
        The name of the best classifier used for analysis.
    region_importance : dict
        A dictionary containing the importance of different brain regions.
    features : list
        A list of features extracted during the analysis.
    plots : list
        A list of URLs or paths to the generated plots for the analysis.
    """

    def __init__(self, accuracy, best_classifier, region_importance, features, plots):
        self.accuracy = accuracy
        self.best_classifier = best_classifier
        self.region_importance = region_importance
        self.features = features
        self.plots = plots

    def to_dict(self):
        """
        Convert the analysis result to a dictionary format for easy serialization.

        Returns:
        --------
        dict
            A dictionary representation of the analysis result.
        """
        return {
            'accuracy': self.accuracy,
            'best_classifier': self.best_classifier,
            'region_importance': self.region_importance,
            'features': self.features,
            'plots': self.plots
        }