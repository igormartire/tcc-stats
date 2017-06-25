import csv

import sys

from collections import Counter

import arff


req_version = (3, 0)
cur_version = sys.version_info

if cur_version < req_version:
    raise Exception("Python 3 is required.")


def main():
    """Print statistics for GO + PPI datasets."""
    arff_files_paths = sys.argv[1:]
    w = csv.DictWriter(sys.stdout, [])
    print_header = True
    for arff_file_path in arff_files_paths:
        dataset = arff.load(arff_file_path)
        dataset_stats = DatasetStats(dataset).get_stats()
        dataset_name = arff_file_path[10:-17]
        dataset_stats['Dataset'] = dataset_name
        if print_header:
            w = csv.DictWriter(sys.stdout, dataset_stats.keys())
            w.writeheader()
            print_header = False
        w.writerow(dataset_stats)


class DatasetStats:
    def __init__(self, dataset):
        self.dataset = dataset
        self._num_positive_instances = None
        self._num_negative_instances = None
        self._num_features = None
        self._num_go_features = None
        self._num_ppi_features = None
        self._rows_stats = None
        self._avg_rows_stats = None

    @property
    def num_instances(self):
        return self.num_positive_instances + self.num_negative_instances

    @property
    def num_positive_instances(self):
        if self._num_positive_instances is None:
            self._calculate()
        return self._num_positive_instances

    @property
    def num_negative_instances(self):
        if self._num_negative_instances is None:
            self._calculate()
        return self._num_negative_instances

    @property
    def num_features(self):
        if self._num_features is None:
            self._calculate()
        return self._num_features

    @property
    def num_go_features(self):
        if self._num_go_features is None:
            self._calculate()
        return self._num_go_features

    @property
    def num_ppi_features(self):
        if self._num_ppi_features is None:
            self._calculate()
        return self._num_ppi_features

    @property
    def rows_stats(self):
        if self._rows_stats is None:
            self._calculate()
        return self._rows_stats

    def _calculate(self):
        self._num_positive_instances = 0
        self._num_negative_instances = 0
        self._rows_stats = []
        for row in self.dataset:
            if row['class'] == '0':
                self._num_negative_instances += 1
            else:
                self._num_positive_instances += 1

            self._rows_stats.append(RowStats(row))
        self._num_features = self._rows_stats[0].num_features
        self._num_go_features = self._rows_stats[0].num_go_features
        self._num_ppi_features = self._rows_stats[0].num_ppi_features
        self._validate()

    def _validate(self):
        assert self.num_instances == (self.num_negative_instances +
                                      self.num_positive_instances)
        for row_stats in self.rows_stats:
            row_stats.validate()
            assert self._num_features == row_stats.num_features
            assert self._num_go_features == row_stats.num_go_features
            assert self._num_ppi_features == row_stats.num_ppi_features

    @property
    def avg_rows_stats(self):
        if self._avg_rows_stats is None:
            self._avg_rows_stats = {}
            summed_rows_stats = Counter()
            for row_stats in self.rows_stats:
                summed_rows_stats += Counter(row_stats.get_stats())

            for key in summed_rows_stats.keys():
                self._avg_rows_stats[key] = (summed_rows_stats[key] /
                                             self.num_instances)
        return self._avg_rows_stats

    def get_stats(self):
        return {
            '#Inst': self.num_instances,
            '#Pos': self.num_positive_instances,
            '#Neg': self.num_negative_instances,
            '%GO': self.num_go_features / self.num_features,
            '%PPI': self.num_ppi_features / self.num_features,
            'avg%GO=0': self.avg_rows_stats['%GO=0'],
            'avg%GO=1': self.avg_rows_stats['%GO=1'],
            'avg%PPI=0': self.avg_rows_stats['%PPI=0'],
            'avg%PPI=(0;150]': self.avg_rows_stats['%PPI=(0;150]'],
            'avg%PPI=(150;400]': self.avg_rows_stats['%PPI=(150;400]'],
            'avg%PPI=(400;700]': self.avg_rows_stats['%PPI=(400;700]'],
            'avg%PPI=(700;900]': self.avg_rows_stats['%PPI=(700;900]'],
            'avg%PPI=(900;1000]': self.avg_rows_stats['%PPI=(900;1000]']
        }


class RowStats:
    def __init__(self, row):
        self.row = row
        self._features_names = {}
        self._values_count = {}

    @property
    def all_features_names(self):
        if 'all' not in self._features_names:
            columns_names = {k for k in self.row._data.keys()
                             if isinstance(k, str)}
            self._features_names['all'] = columns_names - \
                set(['entrez', 'class'])
        return self._features_names['all']

    @property
    def go_features_names(self):
        if 'go' not in self._features_names:
            self._features_names['go'] = {name for name in
                                          self.all_features_names
                                          if name.startswith('GO:')}
        return self._features_names['go']

    @property
    def ppi_features_names(self):
        if 'ppi' not in self._features_names:
            self._features_names['ppi'] = {name for name in
                                           self.all_features_names
                                           if not name.startswith('GO:')}
        return self._features_names['ppi']

    @property
    def num_features(self):
        return len(self.all_features_names)

    @property
    def num_go_features(self):
        return len(self.go_features_names)

    @property
    def num_ppi_features(self):
        return len(self.ppi_features_names)

    @property
    def go_values_counts(self):
        if 'go' not in self._values_count:
            self._values_count['go'] = {
                '0': 0,
                '1': 0
            }
            for go_feature_name in self.go_features_names:
                go_value = self.row[go_feature_name]
                assert go_value in ('0', '1')
                self._values_count['go'][go_value] += 1
        return self._values_count['go']

    @property
    def ppi_values_counts(self):
        if 'ppi' not in self._values_count:
            self._values_count['ppi'] = {
                '0': 0,
                '001->150': 0,
                '151->400': 0,
                '401->700': 0,
                '701->900': 0,
                '901->1000': 0
            }
            for ppi_feature_name in self.ppi_features_names:
                ppi_value = float(self.row[ppi_feature_name])
                if (ppi_value == 0):
                    self._values_count['ppi']['0'] += 1
                elif (0.000 < ppi_value <= 0.150):
                    self._values_count['ppi']['001->150'] += 1
                elif (0.150 < ppi_value <= 0.400):
                    self._values_count['ppi']['151->400'] += 1
                elif (0.400 < ppi_value <= 0.700):
                    self._values_count['ppi']['401->700'] += 1
                elif (0.700 < ppi_value <= 0.900):
                    self._values_count['ppi']['701->900'] += 1
                elif (0.900 < ppi_value <= 1.000):
                    self._values_count['ppi']['901->1000'] += 1
                else:
                    raise Exception('Invalid PPI value for ' +
                                    ppi_feature_name + ' on instance ' +
                                    self.row['entrez'] + '.')
        return self._values_count['ppi']

    def validate(self):
        assert self.num_features == (self.num_go_features +
                                     self.num_ppi_features)
        assert self.num_go_features == sum(self.go_values_counts.values())
        assert self.num_ppi_features == sum(self.ppi_values_counts.values())

    def get_stats(self):
        return {
            '#GO=0': self.go_values_counts['0'],
            '#GO=1': self.go_values_counts['1'],
            '#PPI=0': self.ppi_values_counts['0'],
            '#PPI=(0;150]': self.ppi_values_counts['001->150'],
            '#PPI=(150;400]': self.ppi_values_counts['151->400'],
            '#PPI=(400;700]': self.ppi_values_counts['401->700'],
            '#PPI=(700;900]': self.ppi_values_counts['701->900'],
            '#PPI=(900;1000]': self.ppi_values_counts['901->1000'],
            '%GO=0': self.go_values_counts['0'] / self.num_go_features,
            '%GO=1': self.go_values_counts['1'] / self.num_go_features,
            '%PPI=0': self.ppi_values_counts['0'] / self.num_ppi_features,
            '%PPI=(0;150]': (self.ppi_values_counts['001->150'] /
                             self.num_ppi_features),
            '%PPI=(150;400]': (self.ppi_values_counts['151->400'] /
                               self.num_ppi_features),
            '%PPI=(400;700]': (self.ppi_values_counts['401->700'] /
                               self.num_ppi_features),
            '%PPI=(700;900]': (self.ppi_values_counts['701->900'] /
                               self.num_ppi_features),
            '%PPI=(900;1000]': (self.ppi_values_counts['901->1000'] /
                                self.num_ppi_features)
        }


if __name__ == '__main__':
    main()
