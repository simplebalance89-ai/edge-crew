import os
os.chdir(r"C:\Users\GCTII\AppData\Local\Temp\edge-crew-audit")

with open("server.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the _get_lineup_and_injury_context function with CBS Sports version
old_func_start = 'async def _get_lineup_and_injury_context(sport):'
old_func_end = '    return "\n".join(parts)\n'

start = code.find(old_func_start)
# Find the next function definition after this one
end = code.find('\ndef _build_matrix_section', start)
if end == -1:
    end = code.find('\nasync def _build_matrix_section', start)

if start == -1 or end == -1:
    print(f"FAIL: start={start}, end={end}")
    exit(1)

new_func = '''async def _get_lineup_and_injury_context(sport):
    """Fetch injury data from CBS Sports (server-rendered) + RotoWire lineups."""
    import re
    sport_lower = sport.lower()
    parts = []

    CBS_INJURY_URLS = {
        "nba": "https://www.cbssports.com/nba/injuries/",
        "nhl": "https://www.cbssports.com/nhl/injuries/",
        "soccer": None,
    }

    # --- CBS SPORTS INJURY REPORT (primary - server-rendered, reliable) ---
    cbs_url = CBS_INJURY_URLS.get(sport_lower)
    if cbs_url:
        cache_key = f"cbs_injuries:{sport_lower}"
        cached = _get_cached(cache_key, ttl=1800)
        if cached:
            parts.append(cached)
        else:
            try:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    resp = await client.get(cbs_url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    })
                    if resp.status_code == 200:
                        html = resp.text
                        # CBS structure: <tr> with CellPlayerName, then <td> cells
                        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
                        player_rows = [r for r in rows if 'CellPlayerName' in r]

                        if player_rows:
                            injury_lines = []
                            injury_lines.append(f"INJURY REPORT ({sport.upper()} via CBS Sports - {len(player_rows)} players):")

                            # Also track which teams are affected
                            # CBS groups by team with TableBase-title headers
                            current_team = "Unknown"
                            team_headers = re.findall(
                                r'TableBase-title[^>]*>.*?<a[^>]*>([^<]+)</a>',
                                html, re.DOTALL
                            )

                            for row in player_rows:
                                name_match = re.findall(r'>([^<]+)</a>', row)
                                tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                                cells = []
                                for td in tds:
                                    text = re.sub(r'<[^>]+>', '', td).strip()
                                    if text:
                                        cells.append(text)

                                if name_match and len(cells) >= 4:
                                    # cells: [name_combo, position, date, injury_type, status]
                                    full_name = name_match[-1] if len(name_match) > 1 else name_match[0]
                                    pos = cells[1] if len(cells) > 1 else "?"
                                    injury_type = cells[3] if len(cells) > 3 else "?"
                                    status = cells[4] if len(cells) > 4 else cells[-1]
                                    injury_lines.append(f"  - {full_name.strip()} ({pos}) | {injury_type} | {status}")

                            injury_text = "\n".join(injury_lines)
                            _set_cache(cache_key, injury_text)
                            parts.append(injury_text)
            except Exception as e:
                print(f"CBS injury fetch failed for {sport}: {e}")
                parts.append(f"CBS Sports injury fetch failed: {e}")

    # --- ROTOWIRE LINEUPS (secondary - may be JS-rendered, best effort) ---
    urls = ROTOWIRE_URLS.get(sport_lower, {})
    lineup_url = urls.get("lineups")
    if lineup_url:
        html = await _fetch_rotowire_page(lineup_url, f"rw_lineups:{sport_lower}")
        if html:
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text)
            inj_flags = re.findall(r'(\w[\w\s\.\\'\-]+(?:OUT|GTD|QUESTIONABLE|DOUBTFUL|PROBABLE|DNP))', text)
            if inj_flags:
                parts.append("\nLINEUP PAGE FLAGS (RotoWire):")
                for item in inj_flags[:20]:
                    parts.append(f"  - {item.strip()}")

    if not parts:
        parts.append(f"INJURY/LINEUP: No data sources returned results. Grade conservatively.")

    return "\n".join(parts)

'''

code = code[:start] + new_func + code[end:]

with open("server.py", "w", encoding="utf-8") as f:
    f.write(code)
print(f"OK: Injury source switched to CBS Sports. {code.count(chr(10))} lines.")
