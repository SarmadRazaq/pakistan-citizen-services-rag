# Pakistani Government Services RAG

A multilingual question-answering service over official Pakistani government sources (identity, passport, social protection, tax, rights and complaints) and the user's own uploads.

## Corpus

**Source**:
Something an Agency publishes at a URL, such as a web page or PDF, identified by a seed ID (for example ID-05).
_Avoid_: reference, origin, data source (for a single item)

**Document**:
One file ingested into the system. It is either taken from a Source, or uploaded by a user with no Source behind it.
_Avoid_: file, source

**Version**:
A replacement of a Document's own content. It is still the same Document; a newer edition of a Source is a different Document, related by Supersession.
_Avoid_: edition, revision (when Supersession is meant)

**Translation pair**:
Two Sources that publish the same content in different languages (for example ID-05 in English and ID-06 in Urdu). Citing either one is a full answer, and neither supersedes the other.
_Avoid_: duplicate, Urdu version

**Visibility**:
Whether a Document is public (an official Source, readable by everyone) or private (a User's upload, readable only by its Owner).
_Avoid_: scope (for a Document), shared

**Duplicate**:
A Document with the same file or the same normalised text as one its Owner already has, or as one in the Official corpus. It is refused. Two Users' identical private uploads are not Duplicates of each other, and neither is a Translation pair nor a superseded and current pair.
_Avoid_: copy, re-upload (when a Version is meant)

**Stale**:
A Document whose latest Version replacement failed, so its previous Version is still served until the Owner retries or deletes it.
_Avoid_: broken, outdated (that's Supersession)

**Deletion**:
The immediate, permanent removal of a Document and everything derived from it. There is no trash and no undo.
_Avoid_: archive, soft delete

## People and ownership

**User**:
An account that asks questions and may upload private Documents.
_Avoid_: tenant, customer, member

**Admin**:
A User who may add Documents to the Official corpus.
_Avoid_: superuser, operator

**Owner**:
Whoever a Document belongs to: the User who uploaded it, or the Official corpus for public Documents.
_Avoid_: tenant, uploader, author

**Official corpus**:
The set of all public Documents, taken from Sources and added by Admins.
_Avoid_: seed data, knowledge base, public tenant

## Languages

**Language**:
One of English, Urdu (Urdu script) or Roman Urdu (Urdu written in Latin script). Roman Urdu is a language in its own right, not a script variant of Urdu.
_Avoid_: locale, script (for Roman Urdu)

**Answer language**:
The Language an outcome is written in, always the same as the question's. Quoted passages stay in their Source's Language.
_Avoid_: output language, target language

## Classification

**Agency**:
The government body that publishes a Source, named by its acronym (NADRA, DGIP, FBR, BISP, …).
_Avoid_: department, authority, issuer

**Jurisdiction**:
The level of government a rule applies under: federal, punjab, sindh or kp. Places outside these (Balochistan, ICT, GB, AJK) are outside the corpus and get the federal answer, with a note that provincial rules weren't covered.
_Avoid_: region, province (when federal is included)

**Province**:
A Jurisdiction other than federal: punjab, sindh or kp.
_Avoid_: state, region

**Topic**:
The service area of a Document, from a closed list for public Documents: identity, passport, tax, health, social protection, consumer protection, driving licence, complaints, rights. Optional for private Documents.
_Avoid_: category, subject, tag

**Effective date**:
The date a rule took effect, as best its Source shows it. In order of preference: the issue date in the document's header, then the notification date, then the page's "last updated" date. Never the date it was downloaded or archived; an archive snapshot's date only means "no later than".
_Avoid_: publish date, upload date, created date

## Answering

**Supersession**:
The relationship where a Document is replaced by a newer one on the same subject (same Agency, Jurisdiction and Topic) that has a later Effective date. It is decided when answering, not stored, and applies only between public Documents. Documents with no Effective date are never current or superseded relative to each other.
_Avoid_: outdated, old version, override

**Current**:
The Document, among those on the same subject that are cited in an answer, with the latest Effective date.
_Avoid_: latest, valid, active

**Citation**:
A numbered passage from a Document that an answer is grounded in and points back to.
_Avoid_: source (for a passage), reference, snippet

**Scope**:
Which Documents a question reads from: all (public plus the asker's own), mine (the asker's own only) or public (official Sources only). Scope belongs to a question; Visibility belongs to a Document.
_Avoid_: visibility (for a question), filter

### Outcomes

**Answer**:
An outcome where every part of the question is answered with Citations.
_Avoid_: response, reply

**Partial answer**:
An outcome where some parts of the question are answered with Citations and the rest are named as not covered. It is not a Refusal.
_Avoid_: partial refusal, incomplete answer

**Degraded answer**:
An outcome written by the local fallback model because the hosted model was unavailable. It still carries Citations, but its quality may be lower. It is not a Refusal.
_Avoid_: fallback, error answer

**Refusal**:
An outcome with no answer, for one of these reasons: no documents in Scope, Off-topic question, insufficient context, or blocked by the output guard.
_Avoid_: rejection, decline, "I don't know"

### Question kinds

**Off-topic question**:
A question that has nothing to do with Pakistani government services.
_Avoid_: irrelevant, out of scope (Scope means something else)

**Unanswerable question**:
An on-topic question whose answer is not in the Documents in Scope. It gets a Refusal for insufficient context, never an off-topic one. For a Golden item, it means unanswerable from the Official corpus, whatever Users have uploaded.
_Avoid_: out-of-corpus, impossible question

**Near miss**:
An Unanswerable question where the corpus has a closely related passage that doesn't actually answer it (for example, the NICOP fee for one country when only another's is published). It must never be answered from that passage.
_Avoid_: hard negative, trick question

**Advice request**:
A question about what the asker personally should do, as opposed to what a rule says. It is never refused just for asking: the factual part is answered with Citations, the not-advice Notice is added, and no option is recommended.
_Avoid_: legal advice question, recommendation request

### Notices and referrals

**Notice**:
A fixed message the system attaches to an outcome (not advice, referral, dated information, emergency, third-party processing). It is never written by the model.
_Avoid_: disclaimer, warning, banner

**Referral**:
A verified contact (an office or helpline) that a Notice can carry, matched by Topic and Jurisdiction or by a Referral trigger. A contact that hasn't been checked against its Source is not a Referral and is never shown.
_Avoid_: helpline, contact (when unverified), escalation

**Referral trigger**:
Something found in the question itself, not in a Document's Topic, that calls for a Referral: an emergency or a request for legal advice.
_Avoid_: topic (for emergency or legal advice)

## Evaluation

**Golden item**:
One evaluation question with its expected behaviour: whether it is answerable, its Reference answer, its Gold sources and the Notices it should carry. It is always asked with Scope public.
_Avoid_: test case, sample, eval question

**Reference answer**:
The human-checked answer a Golden item expects, written from its Gold sources.
_Avoid_: ground truth, expected output

**Gold source**:
A Source and page range that a Golden item's answer must be grounded in. It names the Source, not a Document or file, so it survives re-collection.
_Avoid_: gold document, gold file

**Approved**:
A Golden item a named human has checked against its Gold sources. Only Approved items count towards eval results; drafts are candidates and dropped items never run.
_Avoid_: reviewed, validated, final
