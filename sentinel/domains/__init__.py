"""Domain adapters — pluggable monitoring for tabular, time series, NLP, recsys, graph."""

from sentinel.domains.base import BaseDomainAdapter, DomainName

DOMAIN_ADAPTERS: dict[str, str] = {
    "tabular": "sentinel.domains.tabular.adapter:TabularAdapter",
    "timeseries": "sentinel.domains.timeseries.adapter:TimeSeriesAdapter",
    "nlp": "sentinel.domains.nlp.adapter:NLPAdapter",
    "recommendation": "sentinel.domains.recommendation.adapter:RecommendationAdapter",
    "graph": "sentinel.domains.graph.adapter:GraphAdapter",
}


def resolve_adapter(domain: str | DomainName) -> type[BaseDomainAdapter]:
    """Resolve a domain adapter class by name with lazy imports."""
    key = str(domain)
    if key not in DOMAIN_ADAPTERS:
        raise ValueError(f"unknown domain: {key}. Choices: {list(DOMAIN_ADAPTERS)}")
    module_path, class_name = DOMAIN_ADAPTERS[key].split(":")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


__all__ = ["DOMAIN_ADAPTERS", "BaseDomainAdapter", "DomainName", "resolve_adapter"]
