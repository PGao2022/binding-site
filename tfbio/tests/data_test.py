import sys
import os
from glob import glob

import math
import pybel
import numpy as np

import pytest
import tempfile

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(path)[0])

test_dirs = glob(os.path.join(path, 'data', 'complexes', '*'))


def imp_valence(atom):
    return float(atom.OBAtom.GetImplicitValence())


@pytest.fixture(scope='module')
def complexes():
    complexes = []
    for directory in test_dirs:
        pdb_id = os.path.split(directory)[1]
        ligand = next(pybel.readfile('mol2',
                                     os.path.join(directory,
                                                  pdb_id + '_ligand.mol2')))
        pocket = next(pybel.readfile('mol2',
                                     os.path.join(directory,
                                                  pdb_id + '_pocket.mol2')))
        complexes.append((pocket, ligand))
    return complexes


@pytest.fixture(scope='module')
def coords():
    return np.random.rand(1, 3)


def test_encode_num(complexes):
    from tfbio.data import Featurizer
    featurizer = Featurizer()

    for mols in complexes:
        for mol in mols:
            for atom in mol:
                encoding = featurizer.encode_num(atom.atomicnum)
                # one-hot or null
                assert sum(encoding) in [0.0, 1.0]
                if atom.atomicnum in [6, 7, 8]:
                    assert sum(encoding) == 1.0
    # there is no such atomic num
    encoding = featurizer.encode_num(-1)
    assert sum(encoding) == 0.0

    with pytest.raises(TypeError):
        featurizer.encode_num('C')
    with pytest.raises(TypeError):
        featurizer.encode_num(1.0)


def test_find_smarts(complexes):
    from tfbio.data import Featurizer
    featurizer = Featurizer()

    for mols in complexes:
        for mol in mols:
            # at least one feature in the molecule
            smarts = featurizer.find_smarts(mol)
            assert smarts.any()
    with pytest.raises(TypeError):
        featurizer.find_smarts('mol')


def test_get_features(complexes):
    from tfbio.data import Featurizer
    featurizer = Featurizer()

    for mols in complexes:
        for mol in mols:
            coords, features = featurizer.get_features(mol, molcode=1.0)
            assert len(coords) == len(features)

            # at least one feature for each atom
            assert (features != 0).any(axis=1).all()

            with pytest.raises(TypeError):
                featurizer.get_features(mol, 'a')
            with pytest.raises(ValueError):
                featurizer.get_features(mol)

    with pytest.raises(TypeError):
        featurizer.get_features('a')


@pytest.mark.parametrize('axis', ([0, 0, 1], [1, 2, 3]), ids=['Z', 'XYZ'])
@pytest.mark.parametrize('theta1', np.arange(0, 2 * math.pi, 1.75),
                         ids=lambda x: 't1=%s' % x)
@pytest.mark.parametrize('theta2', np.arange(0.5, 2 * math.pi, 2.5),
                         ids=lambda x: 't2=%s' % x)
def test_rotation_matrix(axis, theta1, theta2):
    from tfbio.data import rotation_matrix

    rot11 = rotation_matrix(axis, theta1)
    rot12 = rotation_matrix(axis, theta2)
    rot2 = rotation_matrix(axis, theta1 + theta2)
    assert np.allclose(np.dot(rot11, rot12), rot2)


@pytest.mark.parametrize('axis, theta, err', (
    (1, 0.1, TypeError),
    (['a', 1, 1], 0.1, ValueError),
    ([1, 1], 0.1, ValueError),
    ([1, 1, 1], 'a', TypeError)
), ids=('wrong ax type', 'wrong ax values', 'wrong ax shape', 'wrong theta'))
def test_wrong_rotation_matrix(axis, theta, err):
    from tfbio.data import rotation_matrix
    with pytest.raises(err):
        rotation_matrix(axis, theta)


def test_rotate_int(coords):
    """test predefined rotations"""
    from tfbio.data import rotate, ROTATIONS

    length = np.linalg.norm(coords)
    for rotation in range(len(ROTATIONS)):
        coords_rot = rotate(coords, rotation)
        assert np.allclose(np.linalg.norm(coords_rot), length)


@pytest.mark.parametrize('axis', ([0, 0, 1], [1, 2, 3]), ids=['Z', 'XYZ'])
@pytest.mark.parametrize('theta', np.arange(0, 2 * math.pi, 1.25),
                         ids=lambda x: 't=%s' % x)
def test_rotate_arr(coords, axis, theta):
    """test custom rotation"""
    from tfbio.data import rotate, rotation_matrix

    length = np.linalg.norm(coords)
    rotation = rotation_matrix(axis, theta)
    coords_rot = rotate(coords, rotation)
    assert np.allclose(np.linalg.norm(coords_rot), length)


@pytest.mark.parametrize('rotation, err', (
    (-1, ValueError),
    (100, ValueError),
    (np.random.rand(4, 3), ValueError)
), ids=('negative int', 'int out of bound', 'wrong arr shape'))
def test_wrong_rotation(coords, rotation, err):
    from tfbio.data import rotate

    with pytest.raises(err):
        rotate(coords, rotation)


@pytest.mark.parametrize('crds, err', (
    ('a', TypeError),
    ([['a', 1, 1]], ValueError),
    ([1, 1, 1], ValueError)
), ids=('wrong type', 'wrong value', 'wrong shape'))
def test_wrong_coords(crds, err):
    from tfbio.data import rotate
    with pytest.raises(err):
        rotate(crds, 1)


@pytest.mark.parametrize('dist', (5.0, 10.0, 15.0), ids=lambda x: 'd=%s' % x)
@pytest.mark.parametrize('res', (0.5, 1.0, 2.0), ids=lambda x: 'r=%s' % x)
def test_make_grid(complexes, dist, res):
    from tfbio.data import make_grid, Featurizer
    featurizer = Featurizer()

    # partial charge and molcode can have negative values
    summable_columns = [i for i, f in enumerate(featurizer.FEATURE_NAMES)
                        if f not in ['molcode', 'partialcharge']]

    box_volume = (2 * dist) ** 3

    for mols in complexes:
        for i, mol in zip((-1, 1), mols):
            coords, features = featurizer.get_features(mol, molcode=i)
            coords -= coords.mean(axis=0)

            max_volume = (2 * np.abs(coords).max()) ** 3

            grid = make_grid(coords, features,
                             grid_resolution=res,
                             max_dist=dist)

            # sum values for each feature channel
            sum_grid = np.sum(grid, axis=tuple(range(4)))
            sum_list = np.sum(features, axis=0)
            assert (sum_grid[summable_columns]
                    <= sum_list[summable_columns]).all()
            if max_volume <= box_volume:
                # whole complex fit in the box, so all features
                # should be present in the grid
                assert (np.allclose(sum_grid[summable_columns],
                                    sum_list[summable_columns]))


@pytest.mark.parametrize('crds, fts, kwargs, err', (
    ([['a', 1, 1]], [[1.0]], {}, ValueError),
    ([[1, 1]], [[1.0]], {}, ValueError),
    # ([1, 1], [1.0], {}, ValueError),
    ([[1, 1, 1]], [['a']], {}, ValueError),
    ([[1, 1, 1]], [[1.0], [2.0]], {}, ValueError),
    ([[1, 1, 1]], [[1.0]], {'grid_resolution': 'a'}, TypeError),
    ([[1, 1, 1]], [[1.0]], {'grid_resolution': -1.0}, ValueError),
    ([[1, 1, 1]], [[1.0]], {'max_dist': 'a'}, TypeError),
    ([[1, 1, 1]], [[1.0]], {'max_dist': -1.0}, ValueError),
), ids=('coords value', 'coords shape', 'features value', 'features shape',
        'resolution type', 'resolution value', 'dist type', 'dist value'))
def test_wrong_make_grid(crds, fts, kwargs, err):
    from tfbio.data import make_grid
    with pytest.raises(err):
        make_grid(crds, fts, **kwargs)


def test_cutom_args(complexes):
    from tfbio.data import Featurizer

    featurizer = Featurizer(atom_codes={6: 0, 7: 1, 8: 1},
                            named_properties=['partialcharge'],
                            save_molecule_codes=False,
                            custom_properties=[imp_valence],
                            smarts_properties=['*'])

    labels = ['atom0', 'atom1', 'partialcharge', 'imp_valence', 'smarts0']
    assert featurizer.FEATURE_NAMES == labels

    for mols in complexes:
        for mol in mols:
            coords, features = featurizer.get_features(mol)
            num_heavy_atoms = sum(1 for a in mol if a.atomicnum != 1)
            assert features.shape == (num_heavy_atoms, 5)
            assert (features[:, 3] >= 0).all()
            assert (features[:, 3] <= 5).all()
            assert features[:, 4].all()


@pytest.mark.parametrize('kwargs, err', (
    ({'atom_codes': [(6, 1)]}, TypeError),
    ({'atom_codes': {6: 1}}, ValueError),
    ({'atom_codes': {6: 0}, 'atom_labels': ['C', 'N']}, ValueError),
    ({'named_properties': 'hyb'}, TypeError),
    ({'named_properties': ['a']}, ValueError),
    ({'save_molecule_codes': 'yes'}, TypeError),
    ({'smarts_properties': '*'}, TypeError),
    ({'smarts_properties': ['*'], 'smarts_labels': ['a', 'b']}, ValueError),
    ({'custom_properties': ['hyb']}, TypeError),
), ids=('atom codes list', 'atom codes value', 'to many atom labels',
        'properties dict', 'unknown property', 'save codes', 'SMARTS dict',
        'too many SMARTS labels', 'wrong custom props'))
def test_wrong_featurizer_args(kwargs, err):
    from tfbio.data import Featurizer
    with pytest.raises(err):
        Featurizer(**kwargs)


@pytest.mark.parametrize('kwargs', (
    {'atom_codes': {}},
    {'named_properties': []},
    {'smarts_properties': []}
), ids=('no atom codes', 'no named properties', 'no SMARTS'))
def test_exclude_default_features(complexes, kwargs):
    from tfbio.data import Featurizer

    featurizer = Featurizer(save_molecule_codes=False, **kwargs)
    num_f = len(featurizer.FEATURE_NAMES)
    assert num_f < 18

    for mols in complexes:
        for mol in mols:
            coords, features = featurizer.get_features(mol)
            num_heavy_atoms = sum(1 for a in mol if a.atomicnum != 1)
            assert features.shape == (num_heavy_atoms, num_f)


def test_pickle(complexes):
    from tfbio.data import Featurizer

    featurizer = Featurizer(atom_codes={6: 0, 7: 1, 8: 1},
                            named_properties=['partialcharge'],
                            save_molecule_codes=False,
                            custom_properties=[imp_valence],
                            smarts_properties=['*'])

    with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
        featurizer.to_pickle(f.name)
        featurizer_copy = Featurizer.from_pickle(f.name)

    assert featurizer.FEATURE_NAMES == featurizer_copy.FEATURE_NAMES
    for mols in complexes:
        for mol in mols:
            coords1, features1 = featurizer.get_features(mol)
            coords2, features2 = featurizer_copy.get_features(mol)
            assert np.allclose(coords1, coords2)
            assert np.allclose(features1, features2)
