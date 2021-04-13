# -*- encoding: utf-8 -*-

import setuptools

def read_file(file_name):
    with open(file_name, encoding='utf-8') as fh:
        text = fh.read()
    return text


def get_version(file_name):
    with open(file_name, 'r') as fh:
        for line in fh.readlines():
            if "__version__" in line:
                return line.split()[2].replace('\'', '')

setuptools.setup(
    name='learna',
    author_email='rungef@informatik.uni-freiburg.de',
    description='End-to-end RNA Design using deep reinforcement learning',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    url='https://github.com/automl/learna',
    project_urls={'Source Code': 'https://github.com/automl/learna'},
    version=get_version('learna/__init__.py'),
    packages=setuptools.find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests'],),
    python_requires='>=3',
    platforms=['Linux'],
    classifiers=['Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Development Status :: 4 - Beta',
                 'Natural Language :: English',
                 'Environment :: Console',
                 'Environment :: GPU',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Bio-Informatics']
)
