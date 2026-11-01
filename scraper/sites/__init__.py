"""Register van ondersteunde sites. Nieuwe site: maak sites/<naam>.py en voeg hem hier toe."""

from ..crawler import CareerSiteCrawler
from .kpmg import KPMGCrawler

CRAWLERS: dict[str, type[CareerSiteCrawler]] = {
    "kpmg": KPMGCrawler,
}


def get_crawler_class(site: str) -> type[CareerSiteCrawler]:
    return CRAWLERS.get(site, CareerSiteCrawler)
