"""The grounding document the demo streams back.

EchoLLM answers extractively from `[S1]` style evidence blocks. Supplying a
long document is how the demo gets a stream of a few hundred tokens, which is
what makes time to first token, cadence, mid-stream resume and buffer pressure
visible. With no evidence EchoLLM echoes the prompt and every stream is over
before any of that is observable.

Two constraints on the text, both discovered by reading llmkit/providers.py
rather than guessing: EchoLLM returns the first sentence of each selected
block, so each block is deliberately one long sentence; and a prompt
containing the words judge, grade, score or rate routes to EchoLLM's judge
branch and returns a verdict object instead of prose, so those words are
avoided here.
"""

EVIDENCE = (
    "[S1] A streaming inference server holds one open connection per request and pushes each "
    "decoded delta to the client the moment the model produces it, which means the perceived "
    "latency of the whole interaction collapses to the time needed to produce the very first "
    "token instead of the time needed to produce the last one, and it also means the server now "
    "owns a long lived socket whose reading speed is entirely outside its own control, so every "
    "component sitting between the decoder and the network card needs an explicit answer to the "
    "question of what should happen when the consumer is slower than the producer, a question "
    "that unbounded in memory queues answer badly by quietly converting a slow reader into "
    "resident memory that survives for the entire life of the connection and then multiplies by "
    "however many concurrent streams the process happens to be carrying at that moment\n"
    "[S2] Reconnection is the other half of the same problem because a mobile client will lose "
    "its connection partway through a long generation and a naive client simply reissues the "
    "original request, which restarts generation from the beginning, bills the caller a second "
    "time for tokens already paid for and shows the reader a duplicated prefix, whereas a client "
    "that remembers the identifier of the last event it successfully processed and replays that "
    "identifier in the Last Event ID header when it reconnects lets the server look up the token "
    "sequence it already produced and continue from the exact offset where the previous "
    "connection stopped, so the reader sees one continuous answer and the operator pays once"
)

QUESTION = "what should happen when the consumer is slower than the producer"
