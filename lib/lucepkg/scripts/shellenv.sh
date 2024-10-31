#!/bin/bash

# Function to find the monorepo root by looking for 'lib' and 'project' directories
find_monorepo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/lib" ] && [ -d "$dir/project" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo "Error: Could not find monorepo root." >&2
    return 1
}

# Function to get the path of a package
get_package_path() {
    local package_name="$1"
    local project_path="$MONOREPO_ROOT/project/$package_name"
    local lib_path="$MONOREPO_ROOT/lib/$package_name"

    if [ -d "$project_path" ]; then
        echo "$project_path"
        return 0
    elif [ -d "$lib_path" ]; then
        echo "$lib_path"
        return 0
    else
        echo "Error: Package '$package_name' not found in project/ or lib/ directories." >&2
        return 1
    fi
}

# Function to activate an env and cd there
activate_env() {
    local package_name="$1"

    if [ -n "$package_name" ]; then
        PACKAGE_PATH="$(get_package_path "$package_name")"
        if [ $? -ne 0 ]; then
            echo "$PACKAGE_PATH" >&2
            return 1
        fi
        VENV_PATH="$PACKAGE_PATH/.venv/bin/activate"
        if [ -f "$VENV_PATH" ]; then
            source "$VENV_PATH"
            cd "$PACKAGE_PATH"
        else
            echo "Error: Virtual environment for package '$package_name' not found at $VENV_PATH" >&2
        fi
    else
        # Activate base venv
        BASE_VENV_PATH="$MONOREPO_ROOT/.venv/bin/activate"
        if [ -f "$BASE_VENV_PATH" ]; then
            source "$BASE_VENV_PATH"
            cd "$MONOREPO_ROOT"
        else
            echo "Error: Base virtual environment not found at $BASE_VENV_PATH" >&2
        fi
    fi
}

# Function to install an env
install_env() {
    local dir="$1"
    local do_activate="$2"

    cd "$dir"
    uv sync
    if [ "$dir" = "$MONOREPO_ROOT" ]; then
        .venv/bin/pre-commit install
    fi
    if [ "$do_activate" = true ]; then
        source "$dir/.venv/bin/activate"
    fi
}

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Check if "lucepkg" is in the script directory
if [[ "$SCRIPT_DIR" != *"/lucepkg/"* ]]; then
    echo "Warning: This script is not being run from the expected 'lucepkg' directory. Something might be wrong." >&2
    echo "Current directory: $SCRIPT_DIR" >&2
fi

# Find the monorepo root
MONOREPO_ROOT="$(find_monorepo_root "$SCRIPT_DIR")"
if [ $? -ne 0 ]; then
    echo "Error: Could not find monorepo root." >&2
    return 1
fi

# Define the luce function
luce() {
    local command=""
    local subcommand=""
    local package_name=""
    local force=false
    local all=false
    local port=""
    local node_version=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            activate|remove|install|nb|node|uv)
                command="$1"
                shift
                if [ "$command" = "nb" ] || [ "$command" = "node" ] || [ "$command" = "uv" ]; then
                    subcommand="$1"
                    shift
                fi
                ;;
            --force|-f)
                force=true
                shift
                ;;
            --all|-a)
                all=true
                shift
                ;;
            --port|-p)
                port="$2"
                shift 2
                ;;
            --version|-v)
                node_version="$2"
                shift 2
                ;;
            -*)
                echo "Error: Unknown option $1" >&2
                return 1
                ;;
            *)
                if [ -z "$package_name" ]; then
                    package_name="$1"
                else
                    echo "Error: Unexpected argument $1" >&2
                    return 1
                fi
                shift
                ;;
        esac
    done

    # Validate options for non-install commands
    if [ "$command" != "install" ] && { [ "$force" = true ] || [ "$all" = true ]; }; then
        echo "Error: --force and --all options can only be used with the install command." >&2
        return 1
    fi

    # For luce node install, set default node version if not specified
    if [ "$command" = "node" ] && [ "$subcommand" = "install" ] && [ -z "$node_version" ]; then
        node_version="22"
    fi

    # Execute command
    case "$command" in
        activate)
            activate_env "$package_name"
            ;;
        remove)
            if [ -z "$package_name" ]; then
                echo "Error: Package name is required for remove." >&2
                return 1
            fi
            PACKAGE_PATH="$(get_package_path "$package_name")"
            if [ $? -ne 0 ]; then
                echo "$PACKAGE_PATH" >&2
                return 1
            fi
            VENV_PATH="$PACKAGE_PATH/.venv"

            if [ -d "$VENV_PATH" ]; then
                echo "Removing virtual environment for $package_name..."
                rm -rf "$VENV_PATH"
                if [ $? -ne 0 ]; then
                    echo "Error: Failed to remove virtual environment." >&2
                    return 1
                fi
                echo "Virtual environment for $package_name removed successfully."
            else
                echo "No virtual environment found for $package_name." >&2
            fi
            ;;
        install)
            # Deactivate any active virtual environment if it exists
            if [ -n "$VIRTUAL_ENV" ]; then
                echo "Deactivating current virtual environment..."
                deactivate
            fi
            # Install different packages depending on args
            if $all; then
                # Install all packages
                for dir in "$MONOREPO_ROOT"/lib/* "$MONOREPO_ROOT"/project/*; do
                    echo $dir
                    if [ -d "$dir" ]; then
                        echo "Installing package in $dir"
                        install_env "$dir" false
                    fi
                done
                install_env "$MONOREPO_ROOT" true
            elif [ -z "$package_name" ]; then
                install_env "$MONOREPO_ROOT" true
            else
                PACKAGE_PATH="$(get_package_path "$package_name")"
                if [ $? -ne 0 ]; then
                    echo "$PACKAGE_PATH" >&2
                    return 1
                fi
                install_env "$PACKAGE_PATH" true
            fi
            ;;
        nb)
            case "$subcommand" in
                start)
                    if [ -z "$port" ]; then
                        echo "Error: --port option is required for nb start" >&2
                        return 1
                    fi
                    # Activate base environment
                    activate_env ""
                    # Start Jupyter notebook server in the background and capture PID
                    jupyter notebook --notebook-dir="$MONOREPO_ROOT" --port="$port" > "jupyter_${port}_log_gitignore.txt" 2>&1 & JUPYTER_PID=$!
                    # Wait a bit for the server to start
                    sleep 5
                    # Get and print the token URL
                    echo "Jupyter notebook server started with PID: $JUPYTER_PID"
                    echo "Server logs are being written to: $MONOREPO_ROOT/jupyter_${port}_log_gitignore.txt"
                    jupyter server list | grep ":$port/"
                    ;;
                register)
                    # Activate the package's environment
                    activate_env "$package_name"
                    # Register the kernel
                    .venv/bin/python -m ipykernel install --user --name="$package_name" --display-name="TL Remote: $package_name"
                    echo "Kernel for $package_name registered successfully."
                    ;;
                *)
                    echo "Usage: luce nb <subcommand> [options] [<package_name>]"
                    echo "Subcommands:"
                    echo "  start --port|-p <port>    Start a Jupyter notebook server from base environment"
                    echo "  register <package_name>   Register a kernel for the specified package"
                    echo "Options:"
                    echo "  --port, -p <port>         Specify the port for the Jupyter notebook server (required for start)"
                    ;;
            esac
            ;;
        node)
            case "$subcommand" in
                install)
                    # Check if node is already installed
                    if command -v node &> /dev/null && command -v npm &> /dev/null; then
                        echo "Node.js is already installed:"
                        echo "Node.js version: $(node -v)"
                        echo "npm version: $(npm -v)"
                        echo "Use 'nvm install -v <version>' if you want to switch Node.js versions."
                        return 0
                    fi

                    echo "Installing Node.js environment..."

                    # Check if nvm is already installed
                    if [ -d "$HOME/.nvm" ]; then
                        echo "nvm is already installed"
                    else
                        echo "Installing nvm..."
                        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
                    fi

                    # Source nvm
                    export NVM_DIR="$HOME/.nvm"
                    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

                    # Verify nvm is available
                    if ! command -v nvm &> /dev/null; then
                        echo "Error: nvm installation failed or not properly sourced" >&2
                        echo "Please restart your terminal and try again" >&2
                        return 1
                    fi

                    # Install Node.js with specified version
                    echo "Installing Node.js version $node_version..."
                    nvm install "$node_version"

                    # Verify installation
                    echo "Installation complete!"
                    echo "Node.js version:"
                    node -v
                    echo "npm version:"
                    npm -v
                    ;;
                *)
                    echo "Usage: luce node <subcommand> [options]"
                    echo "Subcommands:"
                    echo "  install [--version|-v <version>]    Install Node.js using nvm (default version: 22)"
                    echo "Options:"
                    echo "  --version, -v <version>          Specify Node.js version to install"
                    ;;
            esac
            ;;
        uv)
            case "$subcommand" in
                install)
                    echo "Installing uv package installer..."
                    curl -LsSf https://astral.sh/uv/install.sh | sh

                    # Source the appropriate RC file based on shell
                    if [ -n "$ZSH_VERSION" ]; then
                        echo "Sourcing ~/.zshrc..."
                        source ~/.zshrc
                    elif [ -n "$BASH_VERSION" ]; then
                        echo "Sourcing ~/.bashrc..."
                        source ~/.bashrc
                    else
                        echo "Warning: Unknown shell type. You may need to restart your shell or manually source your .rc file."
                    fi
                    ;;
                *)
                    echo "Usage: luce uv <subcommand>"
                    echo "Subcommands:"
                    echo "  install    Install the uv package installer"
                    ;;
            esac
            ;;
        *)
            echo "Usage: luce <command> [options] [<package_name>]"
            echo "Commands:"
            echo "  activate [<package_name>]  Activate a virtual environment"
            echo "  remove <package_name>      Remove a virtual environment"
            echo "  install [options] [<package_name>]  Install monorepo package(s) and activate environment"
            echo "  nb <subcommand> [options] [<package_name>]  Notebook-related commands"
            echo "  node <subcommand> [options]          Node.js-related commands"
            echo "  uv <subcommand>           UV package installer commands"
            echo "Options:"
            echo "  --force, -f  Force reinstallation (for install command)"
            echo "  --all        Install all packages (for install command)"
            echo "  --port, -p <port>  Specify the port for Jupyter notebook server (for nb start command)"
            echo "  --version, -v <version>  Specify Node.js version to install (for node install command)"
            ;;
    esac
}

# Export MONOREPO_ROOT in case it's useful elsewhere
export MONOREPO_ROOT
