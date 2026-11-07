# GoodTerms, the 90-second demo

Open the link on the projector. It shows two phones: landlord on the left, renter on the right. Someone from the audience is the renter.

**0:00 Renter, 35 seconds.** Right phone, Start as renter. They type their name. Area, budget, bedrooms. Income screen, leave it or move it. Show my homes. The scan runs: "969,825 London homes, then area, budget, size, and finally the ones whose landlord would accept you. That last number is the product. Ranked by a gradient-boosted model trained on 1.28 million real sales."

**0:35 Landlord, 25 seconds.** Left phone, Start as landlord. Continue. The form fills itself from what the renter just asked for: same area, same bedrooms, a rent they can afford, rules they pass. Photos and description are already in. Scroll to the Guarantee block if you have time. Publish. "Shown first to matching renters near N16."

**1:00 The match, 30 seconds.** Right phone, tap Not for me. The next card is the landlord's home, "Listed by Hassan, just now". "It is first because the renter passes every rule the landlord set. No form, no agent, no referencing fee spent on a home that was never going to say yes." Tap Request a viewing, pick a reason and a time. Left phone, scroll down: the request is there with the ticks. Offer a viewing. Done.

## Before you go on stage

- Reset, so the renter is the only renter and the landlord the only landlord:
  `sqlite3 chairmatch.sqlite "delete from landlords; delete from interests; delete from listing_media; delete from swipes; delete from renters;"` then restart the server.
- The area must be in London. A postcode like S1 (Sheffield) is refused on the area screen with a message, so nobody gets an empty deck.
- If the renter picks 4+ bedrooms the landlord form caps at 3 and the match will not fire. Studio to 3 is safe.
- If the renter sets income very low and has no guarantor, the landlord's rent is capped to what they can afford; the listing may then be far below market. That is fine, say so.
