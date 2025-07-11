"""
Agent Configuration Module.
Uses Pydantic for robust configuration management.
Defines settings for LLMs, tools, database connections, and other agent behaviors.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Attempt to import Pydantic, but proceed if not found for conceptual structure.
try:
    from pydantic import BaseModel, Field, FilePath, DirectoryPath, HttpUrl
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Define BaseModel as a fallback for structure if Pydantic isn't installed
    class BaseModel: # type: ignore
        def __init__(self, **data: Any):
            for key, value in data.items():
                setattr(self, key, value)
        def model_dump_json(self, indent: Optional[int] = None) -> str:
            import json
            return json.dumps(self.__dict__, indent=indent)
        @classmethod
        def model_validate_json(cls, json_str: str) -> 'BaseModel':
            import json
            return cls(**json.loads(json_str))
        # Add other methods as needed for conceptual parity

    # Dummy Field, FilePath etc. for type hinting if Pydantic is not there
    def Field(*args, **kwargs): return None # type: ignore
    FilePath = Path # type: ignore
    DirectoryPath = Path # type: ignore
    HttpUrl = str # type: ignore


logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for a specific LLM."""
    provider: str = Field(default="openai", description="LLM provider (e.g., 'openai', 'huggingface', 'anthropic')")
    model_name: str = Field(description="Name or path of the LLM model.")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider, if applicable.")
    # Example: "OPENAI_API_KEY" -> will attempt to load from env var if api_key is None.
    api_key_env_var: Optional[str] = Field(default=None, description="Environment variable name for the API key.")

    # HuggingFace specific
    hf_auth_token: Optional[str] = Field(default=None, description="HuggingFace auth token.")
    hf_device: Optional[str] = Field(default="auto", description="Device for HuggingFace model ('cpu', 'cuda', 'mps', 'auto').")
    use_4bit: bool = Field(default=False, description="Load HuggingFace model in 4-bit.")
    use_8bit: bool = Field(default=False, description="Load HuggingFace model in 8-bit.")

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Add other common generation params as needed

    class Config:
        extra = "allow" # Allow additional fields not explicitly defined

class ToolConfig(BaseModel):
    """Generic configuration for a tool."""
    name: str
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)

class DatabaseConfig(BaseModel):
    """Configuration for database connection (relevant for benchmark execution)."""
    db_type: str = Field(default="sqlite", description="Type of database (e.g., 'sqlite', 'postgres', 'mysql', 'duckdb').")
    db_path: Optional[str] = Field(default=None, description="Path to SQLite DB file or name of DB for other types.") # For SQLite, this is a file path.
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    # Add other DB specific connection params

class AgentSettings(BaseModel):
    """Main configuration for the SQL Agent."""
    agent_name: str = "NovaSQLAgent"
    default_llm: str = Field(default="default_openai_gpt4o", description="Identifier for the default LLM configuration to use.")
    llm_configs: Dict[str, LLMConfig] = Field(default_factory=dict, description="Dictionary of available LLM configurations.")

    tool_configs: List[ToolConfig] = Field(default_factory=list, description="List of tool configurations.")

    # For Spider/DBT benchmark related settings
    benchmark_name: Optional[str] = Field(default="Spider2-DBT", description="Name of the benchmark being targeted.")
    target_databases: List[DatabaseConfig] = Field(default_factory=list, description="List of database configurations for the benchmark.")
    evaluation_output_dir: DirectoryPath = Field(default_factory=lambda: Path("./evaluation_results"))

    # Paths
    prompt_library_dir: Optional[DirectoryPath] = Field(default_factory=lambda: Path("./prompts"))
    log_level: str = Field(default="INFO", description="Logging level (e.g., DEBUG, INFO, WARNING).")

    class Config:
        extra = "ignore" # Ignore extra fields during parsing

    def get_llm_config(self, name: Optional[str] = None) -> Optional[LLMConfig]:
        """Returns the specified LLMConfig, or the default if name is None."""
        target_name = name if name else self.default_llm
        config = self.llm_configs.get(target_name)
        if not config:
            logger.warning(f"LLM configuration '{target_name}' not found.")
            return None

        # Resolve API key from environment variable if not directly provided
        if config.api_key is None and config.api_key_env_var:
            import os
            env_val = os.getenv(config.api_key_env_var)
            if env_val:
                config.api_key = env_val
                logger.info(f"Loaded API key for LLM '{target_name}' from env var '{config.api_key_env_var}'.")
            else:
                logger.warning(f"API key env var '{config.api_key_env_var}' for LLM '{target_name}' not set.")
        return config

    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Retrieves configuration for a specific tool by name."""
        for tool_conf in self.tool_configs:
            if tool_conf.name == tool_name:
                return tool_conf
        logger.warning(f"Tool configuration for '{tool_name}' not found.")
        return None

    def get_db_config(self, db_name_or_type: str) -> Optional[DatabaseConfig]:
        """Retrieves a database configuration by its name (if named) or type."""
        # This is a simple retrieval. A more complex system might match on unique db_name.
        for db_conf in self.target_databases:
            if db_conf.db_path == db_name_or_type or db_conf.db_type == db_name_or_type: # Basic match
                return db_conf
        logger.warning(f"Database configuration for '{db_name_or_type}' not found.")
        return None


def load_agent_settings(config_path: Union[str, Path] = "config/agent_default_config.json") -> AgentSettings:
    """Loads agent settings from a JSON file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Agent configuration file not found at {path}. Returning default settings.")
        # Create a default config and save it for the user as an example
        default_settings = AgentSettings(
            llm_configs={
                "default_openai_gpt4o": LLMConfig(provider="openai", model_name="gpt-4o", api_key_env_var="OPENAI_API_KEY"),
                "local_llama3_8b": LLMConfig(provider="huggingface", model_name="meta-llama/Meta-Llama-3-8B-Instruct", hf_device="auto", use_4bit=True)
            },
            target_databases=[
                DatabaseConfig(db_type="sqlite", db_path="spider_database/spider.sqlite")
            ],
            tool_configs=[
                ToolConfig(name="sql_query_tool", enabled=True),
                ToolConfig(name="schema_inspector_tool", enabled=True)
            ]
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(default_settings.model_dump_json(indent=2)) # Pydantic v2 method
            logger.info(f"Created a default configuration file at {path}.")
        except Exception as e:
            logger.error(f"Could not write default config file to {path}: {e}")
        return default_settings

    try:
        with open(path, "r") as f:
            json_str = f.read()
        if PYDANTIC_AVAILABLE:
            return AgentSettings.model_validate_json(json_str) # Pydantic v2 method
        else: # Fallback basic parsing
            import json
            return AgentSettings(**json.loads(json_str))

    except Exception as e:
        logger.error(f"Error loading agent settings from {path}: {e}. Returning default settings.", exc_info=True)
        return AgentSettings() # Return default on error

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not PYDANTIC_AVAILABLE:
        logger.warning("Pydantic library not found. Configuration validation and advanced features will be limited.")

    # Example of creating a default config and loading it
    CONFIG_FILE_PATH = Path("temp_agent_config.json")

    # Create a default config for the example
    settings_to_save = AgentSettings(
        default_llm="my_openai",
        llm_configs={
            "my_openai": LLMConfig(provider="openai", model_name="gpt-4o", api_key="sk-dummykey", temperature=0.5),
            "my_hf_model": LLMConfig(provider="huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.2", use_4bit=True)
        },
        tool_configs=[
            ToolConfig(name="database_inspector", enabled=True, settings={"max_tables_to_show": 10}),
            ToolConfig(name="sql_executor", enabled=True, settings={"default_timeout": 30})
        ],
        target_databases=[
            DatabaseConfig(db_type="sqlite", db_path="mydb.sqlite"),
            DatabaseConfig(db_type="postgres", host="localhost", port=5432, user="admin", password="password", db_path="analytics_db")
        ],
        evaluation_output_dir=Path("./my_eval_results"),
        log_level="DEBUG"
    )

    logger.info(f"\n--- Saving example configuration to {CONFIG_FILE_PATH} ---")
    try:
        with open(CONFIG_FILE_PATH, "w") as f:
            f.write(settings_to_save.model_dump_json(indent=2))
        logger.info("Example configuration saved.")
    except Exception as e:
        logger.error(f"Could not save example config: {e}")


    logger.info(f"\n--- Loading configuration from {CONFIG_FILE_PATH} ---")
    loaded_settings = load_agent_settings(CONFIG_FILE_PATH)

    print("\nLoaded Agent Name:", loaded_settings.agent_name)
    print("Default LLM Key:", loaded_settings.default_llm)

    default_llm_conf = loaded_settings.get_llm_config()
    if default_llm_conf:
        print("\nDefault LLM Config:")
        # Depending on Pydantic version, how you pretty print might differ.
        # For conceptual, just printing dict is fine.
        print("  Provider:", default_llm_conf.provider)
        print("  Model Name:", default_llm_conf.model_name)
        print("  API Key:", default_llm_conf.api_key[:10] + "..." if default_llm_conf.api_key else "Not set") # Mask key
        print("  Temperature:", default_llm_conf.temperature)

    hf_llm_conf = loaded_settings.get_llm_config("my_hf_model")
    if hf_llm_conf:
        print("\nHF LLM Config (my_hf_model):")
        print("  Provider:", hf_llm_conf.provider)
        print("  Model Name:", hf_llm_conf.model_name)
        print("  4-bit:", hf_llm_conf.use_4bit)

    db_inspector_tool = loaded_settings.get_tool_config("database_inspector")
    if db_inspector_tool:
        print("\nDatabase Inspector Tool Config:")
        print("  Enabled:", db_inspector_tool.enabled)
        print("  Settings:", db_inspector_tool.settings)

    sqlite_db = loaded_settings.get_db_config("sqlite") # Match by type
    if sqlite_db:
        print("\nSQLite DB Config:")
        print("  Type:", sqlite_db.db_type)
        print("  Path:", sqlite_db.db_path)

    # Clean up the dummy config file
    if CONFIG_FILE_PATH.exists():
        CONFIG_FILE_PATH.unlink()
        logger.info(f"\nCleaned up {CONFIG_FILE_PATH}")

    logger.info("\nConceptual agent_config.py example finished.")

# Ensure __init__.py exists in src/config/
# try:
#     (Path(__file__).parent / "__init__.py").touch(exist_ok=True)
# except NameError:
#     pass
# print("src/config/agent_config.py created.")
