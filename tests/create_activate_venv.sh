#!/usr/bin/env bash

# Create virtual environment for the test
testing_env_dir=${HOME}/.local/python_envs
testing_env=${testing_env_dir}/lambast_testing_venv

# Check if venv already exists
mkdir -p ${testing_env_dir}
if [[ ! -d ${testing_env} ]]; then
    python3 -m venv ${testing_env}
fi

# Activate the environment
source ${testing_env}/bin/activate
if [[ ${?} -ne 0 ]]; then
    echo "Failed to activate environment"
    exit 1
fi

# If given the -c option, uninstall packages
if [[ ${1} ]]; then
    case ${1} in
    -c)
        # Clean every package but lambast
        clean=true
        ;;
    -i)
        install=true
        ;;
    esac
fi

if [[ $clean ]]; then
    python3 -m pip uninstall -yr <(python3 -m pip freeze | sed -nr '/lambast/!p')
fi

# Install lambast
if [[ $install ]]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install mypy
    python3 -m pip install flake8
    python3 -m pip install autopep8
    python3 -m pip install -e $lambast_parent_dir
fi
