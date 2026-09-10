"""A 32-turn support conversation used by the demo and the tests.

Written to have the three properties that make a memory system testable:

  1. A fact stated early (turn 2) that is never repeated and is asked about at
     the end. This is the recall the no-memory baseline fails.
  2. A fact that changes mid-conversation (turn 18), so supersession and
     provenance have something real to resolve.
  3. Enough filler in between that working memory overflows several times and
     compression and eviction actually run.
"""
from __future__ import annotations

from typing import List, Tuple

# The answer the headline recall is looking for, and the query that asks for it.
RECALL_QUERY = "which region is our workspace pinned to?"
RECALL_ANSWER = "eu-west"

# The claim that changes, and both of its values.
CHANGED_SUBJECT = "our production database"
CHANGED_PREDICATE = "is"
OLD_VALUE = "postgres 14"
NEW_VALUE = "postgres 16"

TURNS: List[Tuple[str, str]] = [
    ("assistant", "Hi, I am the Meridian support agent. What are you working on today?"),
    ("user", "We are building a nightly export pipeline. Our workspace is pinned to eu-west and our production database is Postgres 14."),
    ("assistant", "Understood. Are you exporting through the API or a direct database read?"),
    ("user", "Through the API. We use the Python SDK for everything."),
    ("assistant", "The Python SDK retries idempotent requests and honours Retry-After automatically."),
    ("user", "Good. The export runs at two in the morning and takes about forty minutes."),
    ("assistant", "That is a long window. How many requests per minute does it generate at peak?"),
    ("user", "Around 900 at peak, which is above the sustained limit I think."),
    ("assistant", "The sustained default is 600 per minute per workspace, burstable to 1000 for ten seconds."),
    ("user", "So we are over it for most of the run. That explains the 429 responses."),
    ("assistant", "Yes. Rate limits are counted per workspace, so more tokens will not add capacity."),
    ("user", "We already added four extra tokens last month hoping that would help."),
    ("assistant", "That will not have changed anything. Revoking them would reduce your audit noise."),
    ("user", "Noted. Can we raise the sustained limit instead?"),
    ("assistant", "Enterprise plans can be raised to 5000 per minute on request. You are on Growth."),
    ("user", "We are not moving to Enterprise this quarter. What else can we do?"),
    ("assistant", "Spread the export over a longer window and honour Retry-After instead of retrying immediately."),
    ("user", "Makes sense. One correction from earlier: our production database is now Postgres 16."),
    ("assistant", "Thanks, I have updated that. The upgrade does not change the API rate limits either way."),
    ("user", "Right. Does pagination affect how many calls we make?"),
    ("assistant", "List endpoints are cursor paginated with a maximum page size of 200, so larger pages mean fewer calls."),
    ("user", "We are requesting 50 per page at the moment."),
    ("assistant", "Raising that to 200 would cut your request count by roughly a factor of four."),
    ("user", "That alone might bring us under the limit. What about the webhook consumer?"),
    ("assistant", "Webhooks are delivered at least once, so the consumer has to be idempotent on the event id."),
    ("user", "It is, we key on event id already."),
    ("assistant", "Then the remaining risk is the retry storm during the export window."),
    ("user", "We will add jitter to the retry schedule."),
    ("assistant", "Full jitter is the right choice. Fixed backoff reconverges the callers into a second burst."),
    ("user", "Understood. Last thing before I write this up."),
    ("assistant", "Go ahead."),
    ("user", "For the data residency section of the write-up, which region is our workspace pinned to?"),
]


def conversation() -> List[Tuple[str, str]]:
    return list(TURNS)


def turn_text(index: int) -> str:
    """1-based, the way the turns are numbered everywhere else in this project."""
    return TURNS[index - 1][1]
