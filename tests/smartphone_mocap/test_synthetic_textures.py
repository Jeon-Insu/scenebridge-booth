from smartphone_mocap.synthdata.textures import GROUPS, parse_search, pick_maps, search_url


def test_search_url_names_category_and_asks_for_downloads():
    url = search_url("Fabric", 6)
    assert "category=Fabric" in url and "limit=6" in url and "include=downloadData" in url


def test_parse_search_picks_the_requested_resolution():
    payload = {"foundAssets": [{
        "assetId": "Fabric026",
        "downloadFolders": {"default": {"downloadFiletypeCategories": {"zip": {"downloads": [
            {"attribute": "2K-JPG", "downloadLink": "https://x/Fabric026_2K-JPG.zip"},
            {"attribute": "1K-JPG", "downloadLink": "https://x/Fabric026_1K-JPG.zip"},
        ]}}}},
    }]}
    assert parse_search(payload) == [("Fabric026", "https://x/Fabric026_1K-JPG.zip")]


def test_pick_maps_finds_colour_roughness_and_opengl_normal():
    names = ["Fabric026_1K-JPG_Color.jpg", "Fabric026_1K-JPG_NormalDX.jpg",
             "Fabric026_1K-JPG_NormalGL.jpg", "Fabric026_1K-JPG_Roughness.jpg"]
    assert pick_maps(names) == {"color": names[0], "roughness": names[3], "normal": names[2]}


def test_every_group_draws_from_some_category():
    assert set(GROUPS) == {"table", "floor", "wall"} and all(GROUPS.values())
