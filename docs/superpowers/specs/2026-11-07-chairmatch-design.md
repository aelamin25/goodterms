# GoodTerms — design, House London #3, 7 Nov 2026

## One line
A matchmaking app for London renting: renters swipe on real homes, landlords set their rules,
a model pairs the best fits, and a final map shows where matches happened and who got nothing.

## Scope for today (17:00 hard stop)
1. Listings: real London dwellings from the UCL House Price per Square Metre file (33 boroughs),
   one row per dwelling, sampled to ~5,000. Fields: postcode (shown as outward code + sector only),
   borough, type, floor area, rooms, bedrooms (derived), EPC, age band, lat/lon (jittered 150 m),
   asking rent (ONS PIPR borough x bedrooms x type, scaled by floor area vs borough median).
2. Renter profile: budget, bedrooms, work postcode, max commute (straight-line proxy), pets,
   weights for transport / safety / green / energy. Income band and referencing status (self-declared).
3. Landlord rules per listing (generated): income multiple (2.5-3x), referencing required, guarantor
   accepted, pet fee. Lawful rules only.
4. Match = renter swipes right AND renter passes the listing's rules. Match screen + chat stub.
5. Ranking: v1 weighted score; v2 LightGBM trained on simulated personas' swipes, re-ranked with
   the live user's swipe history as features.
6. Market map (stretch): run all simulated renters through matching, show matched vs unmatched
   by MSOA with reason (price / size / place / rules).

## Stack
FastAPI + SQLite (listings, users, swipes, matches). Static HTML/JS swipe page (Tailwind CDN,
pointer-event swipe), Leaflet for the map. Runs locally; one `uvicorn` command.

## Not doing
Real listings, real credit data, policy scenarios, auth, real chat, WhereToBuild in the repo.

## Honesty line for the demo
Homes are real London dwellings; "for rent" and rents are modelled from ONS data; landlords and
training swipes are simulated; live swipes from the room are shown separately.
