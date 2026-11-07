# GoodTerms, the 90-second demo

Open the link on the projector. It shows two phones: landlord on the left, renter on the right. Someone from the audience is the renter.

**0:00 Renter, 35 seconds.** Right phone, Start as renter. Name and email are already there. They type an area (S1 is fine, it becomes SE1), set the budget, keep 2 beds, keep the £80,000 income, tap Non-smoker if they like, Show my homes. The scan runs: "969,825 London homes, then area, budget, size, the ones whose landlord would accept you, and finally how many are listed by a real landlord right now: zero." The scan stops on its numbers. Tap "How we did that" for one screen on the matching: 969,825 real homes, a gradient-boosted model (XGBoost family) trained on 1.28 million real sales, landlord rules applied before anything is shown, ranking from open data. Tap "See my homes". The phone settles on "No landlord has listed a home that fits you yet. The moment one does, it appears here."

**0:35 Landlord, 25 seconds.** Left phone, Start as landlord, Continue. The form fills itself from what the renter just asked for: same area, same bedrooms, a rent they can afford, rules they pass. Photos and description are already in. Publish.

**1:00 The match, 30 seconds.** Say nothing. Within three seconds the renter's phone changes on its own: the home appears, "Listed by Hassan, just now". "It is there because the renter passes every rule the landlord set. No form, no agent, no referencing fee spent on a home that was never going to say yes." Request a viewing, pick a reason and a time. Left phone, scroll down: the request with the ticks. Offer a viewing. Done.

## Before you go on stage

- Reset, so the renter is the only renter and the landlord the only landlord:
  `sqlite3 chairmatch.sqlite "delete from landlords; delete from interests; delete from listing_media; delete from swipes; delete from renters;"` then restart the server.
- Any area works. If someone types a non-London postcode such as S1, the app switches it to the closest London district (SE1) and says so under the field.
- If the renter picks 4+ bedrooms the landlord form caps at 3 and the match will not fire. Studio to 3 is safe.
- If the renter sets income very low and has no guarantor, the landlord's rent is capped to what they can afford; the listing may then be far below market. That is fine, say so.
