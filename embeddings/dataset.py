"""
임베딩 실험 데이터셋.

각 항목은 카테고리(첫 분류)와 아이템(마지막 분류)으로 구성.
검색 시 "{item} {category}" 형태로 쿼리하여 동음이의어/모호성 제거.

예) "야옹이" → "야옹이 Pets" (웹툰 작가 대신 고양이로 검색)
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

PairType = Literal["synonym", "distinct", "irrelevant"]


@dataclass
class InterestItem:
    category: str   # 첫 분류 (Sport, Pets, Food ...) — 클러스터링 기준
    item: str       # 마지막 분류 (Manchester United, Persian Cat ...) — 검색/임베딩 대상

    @property
    def search_query(self) -> str:
        """모호성 제거를 위해 카테고리를 함께 쿼리"""
        return f"{self.item} {self.category}"


def _parse(raw_list: List[str]) -> List[InterestItem]:
    items = []
    for raw in raw_list:
        parts = [p.strip() for p in raw.split(">")]
        items.append(InterestItem(category=parts[0], item=parts[-1]))
    return items


_RAW = [
    "Sport>Football>Premier League>Club>Manchester United",
    "Sport>Basketball>NBA>Team>Los Angeles Lakers",
    "Sport>Olympics>Figure Skating>Athlete>Yuna Kim",
    "Sport>Olympics>Gymnastics>Athlete>Son Yeon-jae",
    "Sport>Tennis>Grand Slam>Tournament>Wimbledon",
    "Sport>Motorsport>Formula 1>Driver>Lewis Hamilton",
    "Sport>Esports>League of Legends>Player>Faker",
    "Sport>Baseball>MLB>Team>New York Yankees",
    "Sport>Golf>PGA Tour>Player>Tiger Woods",
    "Sport>Winter Sports>Skiing>Athlete>Mikaela Shiffrin",

    "Pets>Dogs>Breeds>Dog>Golden Retriever",
    "Pets>Dogs>Food>Brand>Royal Canin",
    "Pets>Cats>Breeds>Cat>Persian Cat",
    "Pets>Cats>Food>Brand>Whiskas",
    "Pets>Fish>Aquarium>Species>Betta Fish",
    "Pets>Birds>Parrots>Species>African Grey Parrot",
    "Pets>Accessories>Toys>Product>KONG Classic",
    "Pets>Healthcare>Veterinary>Clinic>Banfield Pet Hospital",
    "Pets>Dogs>Breeds>Dog>French Bulldog",
    "Pets>Cats>Breeds>Cat>Maine Coon",

    "Vehicles>Cars>Electric>Model>Tesla Model 3",
    "Vehicles>Cars>SUV>Model>Kia Sorento",
    "Vehicles>Cars>Sedan>Model>Hyundai Sonata",
    "Vehicles>Motorcycles>Sport>Model>Yamaha YZF-R1",
    "Vehicles>Motorcycles>Cruiser>Model>Harley-Davidson Street Glide",
    "Vehicles>Trucks>Pickup>Model>Ford F-150",
    "Vehicles>Luxury Cars>Brand>Model>Mercedes-Benz S-Class",
    "Vehicles>Supercars>Brand>Model>Ferrari 488",
    "Vehicles>EV>Hybrid>Model>Toyota Prius",
    "Vehicles>Brands>Automotive>Company>BMW",

    "Game>Video Game>RPG>Title>The Witcher 3: Wild Hunt",
    "Game>Video Game>FPS>Title>Call of Duty: Modern Warfare",
    "Game>Video Game>Battle Royale>Title>Fortnite",
    "Game>Video Game>MOBA>Title>League of Legends",
    "Game>Video Game>Sandbox>Title>Minecraft",
    "Game>Console>PlayStation>Title>The Last of Us",
    "Game>Console>Nintendo>Title>The Legend of Zelda: Breath of the Wild",
    "Game>Mobile Game>Gacha>Title>Genshin Impact",
    "Game>Strategy>RTS>Title>StarCraft II",
    "Game>Sports Game>Football>Title>FIFA 23",

    "Finance>Stocks>Technology>Company>Apple Inc.",
    "Finance>Stocks>Automotive>Company>Tesla Inc.",
    "Finance>Cryptocurrency>Asset>Coin>Bitcoin",
    "Finance>Cryptocurrency>Asset>Coin>Ethereum",
    "Finance>Banking>Global>Institution>JPMorgan Chase",
    "Finance>Investment>ETF>Fund>SPDR S&P 500 ETF",
    "Finance>Insurance>Health>Company>UnitedHealth Group",
    "Finance>Fintech>Payments>Company>PayPal",
    "Finance>Real Estate>REIT>Company>Simon Property Group",
    "Finance>Economy>Index>Market>S&P 500",

    "Food>Snack>Chocolate>Brand>Hershey's Chocolate",
    "Food>Korean Food>Soup>Dish>Kimchi Jjigae",
    "Food>Korean Food>BBQ>Dish>Samgyeopsal",
    "Food>Dessert>Cake>Dish>Nutella Banana Cake",
    "Food>Dessert>Ice Cream>Brand>Ben & Jerry's",
    "Food>Fast Food>Burger>Menu>Big Mac",
    "Food>Italian Food>Pasta>Dish>Spaghetti Carbonara",
    "Food>Japanese Food>Sushi>Dish>Salmon Nigiri",
    "Food>Beverage>Coffee>Drink>Starbucks Latte",
    "Food>Street Food>Korean>Dish>Tteokbokki",

    "Leisure>Travel>City>Destination>Paris",
    "Leisure>Travel>Theme Park>Destination>Disneyland",
    "Leisure>Hobby>Photography>Camera>Canon EOS R5",
    "Leisure>Hobby>Camping>Gear>Snow Peak Tent",
    "Leisure>Entertainment>Movie>Title>Inception",
    "Leisure>Entertainment>OTT>Platform>Netflix",
    "Leisure>Music>Pop>Artist>BTS",
    "Leisure>Reading>Books>Series>Harry Potter",
    "Leisure>Art>Museum>Place>Louvre Museum",
    "Leisure>Outdoor>Hiking>Location>Mount Everest",

    "Child Care>Baby Products>Diapers>Brand>Pampers",
    "Child Care>Baby Products>Stroller>Brand>Bugaboo Fox",
    "Child Care>Education>Early Learning>Program>Kumon",
    "Child Care>Education>Books>Title>Goodnight Moon",
    "Child Care>Health>Pediatrics>Organization>American Academy of Pediatrics",
    "Child Care>Nutrition>Baby Food>Brand>Gerber",
    "Child Care>Entertainment>Cartoon>Title>Peppa Pig",
    "Child Care>Safety>Car Seat>Model>Graco Extend2Fit",
    "Child Care>Clothing>Baby Wear>Brand>Carter's",
    "Child Care>Development>Toys>Product>LEGO Duplo",
]

INTERESTS: List[InterestItem] = _parse(_RAW)


def build_test_pairs(
    interests: List[InterestItem] = None,
    n_intra: int = 5,
    n_inter: int = 5,
    n_irrelevant: int = 3,
    seed: int = 42,
) -> List[Tuple[InterestItem, InterestItem, PairType]]:
    """
    데이터셋에서 테스트 쌍 자동 생성.

    - synonym   (intra-category):  같은 카테고리 → 높은 유사도 기대
    - distinct  (inter-category):  다른 카테고리 → 낮은 유사도 기대
    - irrelevant (cross-domain):   가장 동떨어진 카테고리 쌍

    Args:
        n_intra:     synonym 쌍 수
        n_inter:     distinct 쌍 수
        n_irrelevant: irrelevant 쌍 수
        seed:        재현성을 위한 랜덤 시드

    Returns:
        List of (item_a, item_b, pair_type)
    """
    rng = random.Random(seed)
    if interests is None:
        interests = INTERESTS

    # 카테고리별 그룹핑
    by_category: Dict[str, List[InterestItem]] = defaultdict(list)
    for item in interests:
        by_category[item.category].append(item)

    categories = list(by_category.keys())
    pairs: List[Tuple[InterestItem, InterestItem, PairType]] = []

    # synonym: 같은 카테고리에서 2개 샘플
    cats_with_2plus = [c for c in categories if len(by_category[c]) >= 2]
    for _ in range(n_intra):
        cat = rng.choice(cats_with_2plus)
        a, b = rng.sample(by_category[cat], 2)
        pairs.append((a, b, "synonym"))

    # distinct: 다른 카테고리에서 1개씩
    for _ in range(n_inter):
        cat_a, cat_b = rng.sample(categories, 2)
        a = rng.choice(by_category[cat_a])
        b = rng.choice(by_category[cat_b])
        pairs.append((a, b, "distinct"))

    # irrelevant: 카테고리 정렬 후 양 끝에서 샘플 (가장 동떨어진 쌍)
    sorted_cats = sorted(categories)
    half = len(sorted_cats) // 2
    far_pairs = list(zip(sorted_cats[:half], sorted_cats[half:]))
    for _ in range(n_irrelevant):
        cat_a, cat_b = rng.choice(far_pairs)
        a = rng.choice(by_category[cat_a])
        b = rng.choice(by_category[cat_b])
        pairs.append((a, b, "irrelevant"))

    return pairs
