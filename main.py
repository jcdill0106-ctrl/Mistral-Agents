import io
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

MAX_ROWS_RETURNED = 100
MAX_GROUPS_RETURNED = 200

_DATASETS: dict[str, pd.DataFrame] = {}


def _get_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in _DATASETS:
        raise agents.ToolError(
            f"No dataset named '{dataset_name}' is loaded. Call load_csv first."
        )
    return _DATASETS[dataset_name]


class LoadCsvInput(BaseModel):
    dataset_name: Annotated[
        str,
        Field(description="Short name to refer to this dataset in later tool calls, e.g. the file name without extension."),
    ]
    csv_content: Annotated[
        str,
        Field(description="The full raw text content of the CSV file, exactly as downloaded from Google Drive."),
    ]


@agents.tool(
    name="load_csv",
    description=(
        "Load raw CSV text into memory under a dataset name. Call this once per file, "
        "after fetching the file content with the Google Drive connector. Returns the shape, "
        "columns, dtypes and a preview of the data."
    ),
    input_schema=LoadCsvInput,
    model_access="direct",
)
def load_csv(args: LoadCsvInput) -> dict:
    """Load a CSV dataset into memory."""
    try:
        df = pd.read_csv(io.StringIO(args.csv_content), sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(args.csv_content), sep=";", engine="python")
        except Exception as exc:
            raise agents.ToolError(f"Could not parse the content as CSV: {exc}")
    if df.empty:
        raise agents.ToolError("The parsed CSV has no rows.")
    _DATASETS[args.dataset_name] = df
    return {
        "dataset_name": args.dataset_name,
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "preview": df.head(5).to_dict(orient="records"),
    }


class DescribeColumnsInput(BaseModel):
    dataset_name: Annotated[str, Field(description="Name given to load_csv.")]
    column: Annotated[
        str | None,
        Field(description="Optional single column to describe in detail. Omit to describe all columns."),
    ] = None


@agents.tool(
    name="describe_columns",
    description=(
        "Describe the columns of a loaded dataset: dtype, null counts, unique counts, "
        "and for numeric columns min/max/mean/median/std, for text columns the most frequent values."
    ),
    input_schema=DescribeColumnsInput,
    model_access="direct",
)
def describe_columns(args: DescribeColumnsInput) -> dict:
    """Describe one or all columns of a dataset."""
    df = _get_dataset(args.dataset_name)
    cols = [args.column] if args.column else list(df.columns)
    if args.column and args.column not in df.columns:
        raise agents.ToolError(
            f"Column '{args.column}' not found. Available: {list(df.columns)}"
        )
    description: dict[str, dict] = {}
    for col in cols:
        series = df[col]
        info: dict = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            info["min"] = _jsonable(series.min())
            info["max"] = _jsonable(series.max())
            info["mean"] = _jsonable(series.mean())
            info["median"] = _jsonable(series.median())
            info["std"] = _jsonable(series.std())
        else:
            info["top_values"] = (
                series.value_counts(dropna=True).head(5).to_dict()
            )
        description[str(col)] = info
    return {"dataset_name": args.dataset_name, "columns": description}


Operator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]


class FilterRowsInput(BaseModel):
    dataset_name: Annotated[str, Field(description="Name given to load_csv.")]
    column: Annotated[str, Field(description="Column to filter on.")]
    operator: Annotated[
        Operator,
        Field(description="Comparison operator: eq, ne, gt, gte, lt, lte, contains, in."),
    ]
    value: Annotated[
        str | float | int | list,
        Field(description="Value to compare against. A list for the 'in' operator."),
    ]
    limit: Annotated[
        int,
        Field(description=f"Maximum number of rows to return, at most {MAX_ROWS_RETURNED}."),
    ] = 50


@agents.tool(
    name="filter_rows",
    description=(
        "Filter rows of a loaded dataset on one column with a fixed comparison operator. "
        "Returns the total number of matching rows and up to `limit` matching rows."
    ),
    input_schema=FilterRowsInput,
    model_access="direct",
)
def filter_rows(args: FilterRowsInput) -> dict:
    """Filter rows of a dataset with a whitelisted operator."""
    df = _get_dataset(args.dataset_name)
    if args.column not in df.columns:
        raise agents.ToolError(
            f"Column '{args.column}' not found. Available: {list(df.columns)}"
        )
    series = df[args.column]
    value = args.value
    op = args.operator
    if op == "in":
        if not isinstance(value, list):
            raise agents.ToolError("Operator 'in' requires a list of values.")
        mask = series.isin(value)
    elif op == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    else:
        coerced = _coerce(value, series)
        if op == "eq":
            mask = series == coerced
        elif op == "ne":
            mask = series != coerced
        elif op == "gt":
            mask = series > coerced
        elif op == "gte":
            mask = series >= coerced
        elif op == "lt":
            mask = series < coerced
        else:
            mask = series <= coerced
    matched = df[mask]
    limit = max(1, min(int(args.limit), MAX_ROWS_RETURNED))
    return {
        "dataset_name": args.dataset_name,
        "total_matches": int(len(matched)),
        "returned": int(min(len(matched), limit)),
        "rows": matched.head(limit).to_dict(orient="records"),
    }


AggregationFunction = Literal["count", "sum", "mean", "min", "max", "median", "std", "nunique"]


class Aggregation(BaseModel):
    column: Annotated[str, Field(description="Column to aggregate.")]
    function: Annotated[
        AggregationFunction,
        Field(description="Aggregation function: count, sum, mean, min, max, median, std, nunique."),
    ]


class AggregateByGroupInput(BaseModel):
    dataset_name: Annotated[str, Field(description="Name given to load_csv.")]
    group_by: Annotated[
        str | list[str],
        Field(description="One column name, or a list of column names, to group by."),
    ]
    aggregations: Annotated[
        list[Aggregation],
        Field(description="Aggregations to compute per group, e.g. [{'column': 'amount', 'function': 'sum'}]."),
    ]


@agents.tool(
    name="aggregate_by_group",
    description=(
        "Compute aggregations (count, sum, mean, min, max, median, std, nunique) of one or more "
        "columns, grouped by one or more columns of a loaded dataset. Returns one row per group."
    ),
    input_schema=AggregateByGroupInput,
    model_access="direct",
)
def aggregate_by_group(args: AggregateByGroupInput) -> dict:
    """Group a dataset and aggregate with whitelisted functions."""
    df = _get_dataset(args.dataset_name)
    groups = [args.group_by] if isinstance(args.group_by, str) else list(args.group_by)
    for g in groups:
        if g not in df.columns:
            raise agents.ToolError(
                f"Group column '{g}' not found. Available: {list(df.columns)}"
            )
    agg_map: dict[str, list[str]] = {}
    for agg in args.aggregations:
        if agg.column not in df.columns:
            raise agents.ToolError(
                f"Column '{agg.column}' not found. Available: {list(df.columns)}"
            )
        agg_map.setdefault(agg.column, [])
        if agg.function not in agg_map[agg.column]:
            agg_map[agg.column].append(agg.function)
    result = df.groupby(groups, dropna=False).agg(agg_map)
    result = result.reset_index()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(part) for part in col if str(part)) for col in result.columns
        ]
    if len(result) > MAX_GROUPS_RETURNED:
        truncated = result.head(MAX_GROUPS_RETURNED)
    else:
        truncated = result
    return {
        "dataset_name": args.dataset_name,
        "group_by": groups,
        "total_groups": int(len(result)),
        "returned_groups": int(len(truncated)),
        "rows": _records(truncated),
    }


def _coerce(value, series: pd.Series):
    if pd.api.types.is_numeric_dtype(series) and not isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise agents.ToolError(
                f"Column '{series.name}' is numeric; provide a number, got '{value}'."
            )
    if pd.api.types.is_bool_dtype(series):
        return str(value).lower() in ("true", "1", "yes")
    return value


def _jsonable(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    return [{str(k): _jsonable(v) for k, v in row.items()} for row in records]


INSTRUCTIONS = """You are a conversational data analyst. The user gives you a link to a CSV file in their
Google Drive and asks questions about it in natural language (French or English).

Workflow:
1. Use the google_drive_mcp connector to locate the file from the link or name the user gives,
   then download its text content.
2. Call load_csv once with the file content and a dataset name (the file name without extension).
   Remember the dataset name: the whole conversation and all follow-up questions refer to it.
3. Answer every question with load_csv / describe_columns / filter_rows / aggregate_by_group only.
   Never invent numbers: every figure in your answer must come from a tool result.
4. When a question needs filtering then aggregation, chain the tools in that order.

Rules:
- You have no code execution and no shell: use only your four fixed tools. Do not attempt to write code.
- Be concise, answer in the user's language, and show the real figures with their units.
- The session keeps its memory: the user can ask follow-up questions without re-explaining the
  file; the dataset stays loaded for the whole conversation.
"""

agent = agents.Agent(
    model="mistral-large-latest",
    name="csv-analyst",
    instructions=INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, aggregate_by_group],
    connectors=[agents.connector("google_drive_mcp")],
)
