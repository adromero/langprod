# Domain Analysis: Brand Message Coherence Methodology

## 1. Domain Context

### How brand managers think about "messaging coherence"

Brand coherence is not a single construct in practice. Brand managers operate with at least four distinct notions that they conflate under the umbrella of "consistency":

1. **Claims consistency** — Are the factual claims (SPF 50, 12-hour protection, zero sugar) present and accurate across channels? This is the most auditable dimension. Regulated industries (pharma OTC, food, personal care) already have compliance teams checking this, but they do it manually and typically only audit brand-controlled channels.

2. **Voice/tone consistency** — Does the brand sound like itself? A brand that is "scientific authority" on its website but "irreverent bro" on TikTok has a voice coherence problem. This is the dimension brand teams worry about most, and it is fundamentally about register — the very signal that dominates the LLM representations. This creates a paradox: the methodology's biggest "nuisance variable" is the dimension brand managers care most about.

3. **Positioning consistency** — Is the product occupying the same competitive position across channels? A toothpaste positioned as "premium sensitivity relief" on its website but appearing in "budget whitening" Amazon search results has a positioning drift. This is a semantic-structural concept that goes beyond individual claims.

4. **Visual/experiential consistency** — Color, typography, imagery, packaging. Entirely outside the scope of text-based analysis, but brand managers often mean this when they say "coherence." The methodology needs to clearly scope itself to textual/verbal coherence and explain what it does not cover.

The proposed methodology maps best to (1) claims consistency and partially to (3) positioning consistency. It does not measure (2) voice/tone in the way brand managers expect (because that would require measuring register similarity, which is the opposite of what the contrastive fine-tuning does), and it cannot address (4) at all.

**Critical implication for the contrastive fine-tuning decision (Decision #3):** Training a model to collapse register is training it to ignore voice/tone differences. This is correct for measuring "does the same core product message survive translation," but it will not detect a brand that sounds professional on its website and juvenile on social media. The commercial positioning must be very precise about what "coherence" means in this context — semantic content coherence, not voice coherence.

### What $25-75K buys in market research

For context on pricing:

- **Brand tracking studies** (Kantar, Ipsos): $50-150K/year for ongoing quantitative tracking. Quarterly reports on awareness, consideration, brand health KPIs. Large sample sizes (n=500-2000 per wave).
- **Ad testing / copy testing** (System1, Dynata panels): $10-25K per ad, testing emotional response, persuasion, distinctiveness. Single-stimulus focus.
- **Custom qualitative research** (in-depth interviews, focus groups): $25-50K for a multi-market study. Subjective, exploratory, small sample.
- **Brand audit / architecture engagements** (consultancies like Interbrand, Landor): $50-200K. Strategic, not purely quantitative. Deliverable is a positioning recommendation, not just a measurement.
- **Social listening / brand monitoring** (Brandwatch, Sprinklr): $30-80K/year for platform licenses. Ongoing, dashboard-driven. Measures volume, sentiment, topic share — not cross-channel coherence.

The proposed $25-75K range positions this as a one-time custom research engagement, comparable to a brand audit but with a quantitative core. This is plausible if the deliverable includes strategic recommendations, not just numbers. A brand team will not pay $50K for a dashboard — they will pay $50K for a report that tells them "your Amazon listings have drifted from your core positioning, here's what to fix, and here's how you compare to competitors."

### Competitive landscape: what existing tools actually do

**Brandwatch / Sprinklr / Meltwater:** These are social listening platforms. They track brand mentions, sentiment, topics, and share of voice. They do not measure cross-channel coherence of the brand's own messaging. They measure what people say *about* the brand, not whether the brand says the same thing across channels.

**BrandGuard / Frontify / Bynder:** These are brand management / digital asset management platforms. They enforce visual brand guidelines (logo usage, colors, fonts). Some have basic text analysis (keyword lists, approved/prohibited terms). They are compliance tools, not measurement tools. They prevent deviations but do not audit existing messaging.

**Qualtrics BrandXM / Morning Consult:** Consumer perception surveys. They measure how the brand is perceived, not what the brand says. There is a causal chain (inconsistent messaging -> confused perception) but these tools measure the downstream effect, not the messaging itself.

**Persado / Phrasee / Jasper:** AI-powered copywriting and optimization. They help write better copy, not audit existing copy for coherence.

**Gap in the market:** No existing tool quantitatively measures the semantic coherence of a brand's own messaging *across* channels from the brand's own materials. This is the genuine whitespace. Compliance teams check regulatory accuracy. Social listening tracks external conversation. Brand management platforms enforce visual guidelines. But nobody systematically answers: "Does our product website, our Amazon listing, our regulatory filing, and our social media all communicate the same core product story?"

This is a real gap, and brand managers recognize it qualitatively. They conduct ad-hoc "message audits" (typically a junior strategist reading everything and writing a PowerPoint). The proposed methodology replaces that with a quantitative, reproducible process.

## 2. User Workflows

### Who buys this engagement

**Primary buyer: VP of Brand / Brand Director at a CPG company.**
- Typically reports to CMO or GM of a business unit.
- Budget authority: $25-75K is within discretionary brand research budgets (no CFO approval needed at large CPGs).
- Trigger events: brand refresh, agency change, category expansion, M&A integration, launch of a new channel (e.g., DTC), negative consumer feedback about "confusing messaging."
- Decision cycle: 2-4 weeks from initial conversation to signed SOW.

**Secondary buyer: Agency strategist (brand strategy or media agency).**
- Uses the audit to support a pitch ("we audited your competitor, here's what we found") or to justify their own strategic recommendations.
- Lower price sensitivity (agency marks it up to the client).
- Values speed and competitive benchmarking.

**Tertiary buyer: Regulatory/compliance team at a pharma or food company.**
- Cares specifically about claims consistency. "Does our marketing say something our regulatory filing doesn't support?"
- Very different framing: not "coherence" but "compliance risk."
- Could be a distinct product variant.

### What they expect from the engagement

Based on how brand audits work in practice, the engagement would follow this workflow:

1. **Scoping call (Week 0):** Define which products, which channels, which time period. Client identifies 3-10 products and 4-6 channels. Agree on what "coherence" means for their context.

2. **Document collection (Weeks 1-2):** Researcher collects documents. For brand-controlled channels, the client provides materials. For third-party channels (Amazon, consumer reviews, social media), the researcher collects via scraping or APIs. This is the most labor-intensive phase.

3. **Analysis (Weeks 2-3):** Run the extraction pipeline, compute coherence scores, generate pairwise channel matrices, identify outliers. Internal QA.

4. **Report delivery (Week 3-4):** Written report with:
   - Portfolio-level coherence scorecard (one number per product, benchmarked)
   - Channel-level heat map (which channels align, which diverge)
   - Product-specific deep dives for the most problematic products
   - Competitive comparison (if scoped)
   - Recommendations (prioritized list of what to fix)

5. **Presentation (Week 4):** Walk the client through findings. Answer questions. This is where the $50K is justified — not by the analysis, but by the interpretation and strategic framing.

### What would make them say "this is worth $50K"

- **Specificity:** Not "your messaging is somewhat inconsistent" but "your Amazon listing for Product X has drifted 2.3 standard deviations from your core positioning, specifically on the efficacy claim, while your TikTok content is aligned."
- **Actionability:** "Here are the three documents that need revision, here is what they should say instead."
- **Competitive context:** "Your coherence score is 0.72; category average is 0.65; your main competitor is 0.81."
- **Surprise insight:** Finding something the client didn't know. "Your regulatory filing and your consumer reviews are actually more aligned than your marketing copy — your marketing team has drifted from the product's true story."
- **Reproducibility:** "We can re-run this in 6 months to measure improvement." (This creates recurring revenue.)

## 3. Business Logic Verification

### Are the experiments testing the right things?

**Experiment 0 (metric exploration on calibration data):** Correct and necessary. The critique's insistence on pre-registering the metric before Experiment 1 is exactly right. This exploratory phase maps to how research firms develop new methodologies — you calibrate on known data before applying to the wild.

**Experiment 1 (real-document sensitivity):** The right question, but the operationalization needs refinement from a domain perspective. The decision to increase to n=10 per group (Decision #8) helps. However:
- The categories of "known consistent" and "known inconsistent" are too binary. In practice, coherence is a continuum. A better design: select 20 products across a range of expected coherence (from highly controlled OTC pharmaceutical brands to chaotic DTC startups) and test whether the metric correlates with expert rankings. This avoids the artificial binary and is more representative of real use cases.
- The baselines (Decision #5: TF-IDF, BERTScore) are essential for commercial credibility. A client will ask "why not just use ChatGPT to compare my documents?" The response needs to be grounded in empirical comparison.

**Experiment 2 (channel attribution):** Directly maps to the most valuable deliverable — telling the client *which* channel is the problem. This is what separates a $25K report from a $50K report. The test is well-designed from a domain perspective.

**Experiment 3 (attribute drill-down):** The most commercially valuable experiment but also the most technically risky. Brand managers think in terms of specific claims and attributes, not holistic similarity scores. If this works, it is the "killer feature." If it fails, the offering is still viable but positioned differently (scorecard vs. diagnostic). The critique's concern about keyword-matching dominating this test is valid and must be addressed in the probe design.

**Experiment 4 (temporal drift):** Nice-to-have for the initial offering but not essential. Most brand audit engagements are point-in-time. Temporal tracking becomes relevant for recurring engagements. Recommend deprioritizing until after initial commercial validation.

**Experiment 5 (competitive benchmarking):** Essential for commercial positioning. "You score X, your competitor scores Y" is the single most compelling slide in a client presentation. Expert validation of face validity is the right approach, but the critique's point about forced-choice design is well-taken.

### Pass/fail criteria from a market perspective

The current pass/fail criteria are calibrated for scientific rigor (no overlap, correct identification). From a market perspective, the bar is both higher and lower:

- **Lower bar:** A brand manager will accept a tool that is "directionally correct" even if imperfect, as long as the insights are actionable. If the metric correctly identifies 7 of 10 products' relative coherence, that is commercially viable.
- **Higher bar:** The metric must never produce a result that is embarrassingly wrong — i.e., ranking a clearly chaotic brand as highly coherent. One obviously wrong result destroys credibility faster than three slightly-off results.

Recommendation: add a "face validity" check to Experiment 1. After computing scores, present the full ranking (not just the binary grouping) to 2-3 industry professionals. Ask: "Does anything here look obviously wrong?" This catches the credibility-destroying outliers.

## 4. Domain-Specific Requirements

### Document collection specification

Real-world document collection is the operational bottleneck, and the plan underspecifies it. Requirements:

- **Brand-controlled channels:** Product website (primary product page, not the homepage), product packaging text (transcribed), advertising copy (most recent campaign), press releases. These are authored by the brand and represent intentional messaging.
- **Platform-mediated channels:** Amazon/Walmart product listings (often written by the brand but constrained by platform templates), app store descriptions, Google Shopping descriptions. These are brand-authored but platform-influenced.
- **Third-party channels:** Consumer reviews (curated sample, not all reviews), press/media coverage, social media mentions. These are NOT brand-authored and should be analyzed separately per Decision #7 (two-tier reporting).
- **Regulatory channels:** FDA filings (OTC drug facts panels), FTC disclosure documents, ingredient lists, nutrition facts panels. These are brand-authored but heavily constrained by regulation.

Each document needs metadata: source URL, date collected, author type (brand/platform/third-party), word count, channel category.

### Document preprocessing requirements

The plan does not address preprocessing, but domain requirements demand it:

- **Boilerplate removal:** Amazon listings contain template text ("About this item", navigation elements, "Customers also viewed"). Websites contain footer/header/cookie notices. These must be stripped.
- **Multi-product pages:** A landing page for "Colgate Total" might also mention "Colgate Optic White." Need a strategy for isolating product-specific content.
- **Review aggregation:** A single consumer review is noisy. A curated sample of 5-10 reviews, concatenated or individually analyzed, provides a more stable signal.
- **Length normalization:** Per the critique, this is critical. Proposal: extract the "core description" segment (typically 50-200 words) from each document rather than using the full document. For short documents (tweets), use the full text. For long documents (regulatory filings), extract the product description section only.

### Scoring and reporting requirements

For commercial viability, the coherence score needs:

- **Interpretable scale:** Not a raw cosine similarity or RSA r-value. Propose a 0-100 "Brand Coherence Index" (BCI) calibrated against the calibration dataset. 0 = "messaging across channels is completely unrelated." 100 = "messaging is identical in semantic content across all channels." Calibration: the mean within-product coherence from the 800-stimulus dataset (where all information is present in every register) represents the theoretical maximum.
- **Channel-pair decomposition:** A matrix showing coherence between every pair of channels. This is the most useful diagnostic.
- **Attribute-level drill-down (if Experiment 3 passes):** For each key product claim/attribute, a presence/absence indicator per channel.
- **Competitive context (if Experiment 5 passes):** Percentile rank within category.

## 5. Edge Cases

### Multi-product brands (P&G, Unilever)

A single parent company may have 50+ brands. The methodology applies at the individual brand/product level, not the corporate level. However:
- Some channels (corporate website, annual report, press releases) discuss multiple products. Need a product-isolation strategy.
- "Brand architecture" coherence (is Tide positioned consistently relative to Gain?) is a different question than product coherence but may be a commercially interesting extension.
- Corporate brands that appear on product packaging (e.g., "A P&G Product") create a secondary coherence dimension.

### Co-branded products

Products like "Starbucks x Nespresso capsules" or "Nike x Off-White" have inherently multi-voiced messaging. Each brand contributes its own voice. Coherence measurement needs to specify: coherence of the co-brand messaging as a whole, or coherence of each brand's contribution? This is a scoping question for the engagement.

### International variations

The same product may have different formulations, claims, and messaging in different markets. "Colgate Total" in the US vs. UK vs. India is effectively three different products from a coherence standpoint. The methodology should treat each market as a separate study. However, a "global coherence" metric (how consistent is the brand across markets for the same product?) could be a high-value extension for global CPG companies. This would require multilingual embedding support — feasible with multilingual LLMs but not addressed in the current plan.

### Private label / store brands

Store brands (Kirkland Signature, Amazon Basics, Great Value) have minimal brand-controlled messaging. Their "marketing" is essentially the product listing. There may be no website, no social media, and limited regulatory filing differentiation. The methodology would have very few channels to compare, potentially producing meaningless coherence scores (high coherence by default because there is little messaging to diverge). Not a viable target market segment for the initial offering.

### Products undergoing rebranding

A brand in mid-rebrand (e.g., Facebook -> Meta, Weight Watchers -> WW) will intentionally have inconsistent messaging: old messaging on some channels, new messaging on others. The metric would correctly identify this as incoherence, but the client already knows. The value-add here is measuring rebranding *completeness*: "Your website and social media have adopted the new messaging, but your Amazon listings and 73% of consumer reviews still reference the old positioning." This is actually a high-value use case — rebranding executives want to track rollout progress.

### Seasonal / limited edition messaging

Holiday flavors, seasonal campaigns, and limited editions create temporary messaging layers. "Pumpkin Spice Cheerios" has seasonal messaging that is *intentionally* different from "Original Cheerios." The methodology needs to distinguish between intentional messaging variation (seasonal, campaign-specific) and unintentional drift. Recommendation: scope the engagement to "core product messaging" and exclude explicitly seasonal/promotional content. This is a document curation decision, not a metric design decision.

### Products with negative press / recalls

A product involved in a recall, lawsuit, or safety controversy will have a large corpus of negative third-party content that diverges from brand messaging. Under the two-tier model (Decision #7), this would show up as a gap between "brand coherence" (still high, because the brand controls its own messaging) and "market coherence" (low, because the market is saying something different). This is actually useful and differentiating: "Your brand messaging is internally consistent, but the market narrative has diverged significantly due to the recall."

### Products with minimal online presence

Artisanal, local, or niche products may lack sufficient channel diversity for a meaningful audit. Minimum viable analysis requires at least 3 distinct channels with substantive text. This sets a lower bound on which products can be studied.

## 6. Standards & Compliance

### Data collection ethics

- Consumer reviews are publicly posted content, generally legal to scrape and analyze. However, some platforms (Amazon, Yelp) have ToS restrictions on systematic scraping. The engagement should use API access where available and comply with robots.txt.
- Brand materials collected from public sources (websites, listings) are factual business documents. No PII concerns.
- Client-provided materials (internal briefs, pre-launch copy) may be confidential. The engagement needs an NDA and clear data handling provisions. The LLM extraction pipeline must not send client materials to external APIs — this favors local model inference (as the current pipeline uses with Qwen2.5-32B on the RTX 5090).

### Research methodology standards

If positioning this as "quantitative research," the methodology should adhere to market research industry norms:
- **ESOMAR / Insights Association guidelines** for research transparency.
- **Pre-registration** of the metric definition before applying it to client data (aligns with Decision #4 on Experiment 0).
- **Reproducibility:** Same inputs should produce same outputs. The pipeline must be deterministic (set random seeds, fix model versions, document preprocessing steps).
- **Limitations disclosure:** The report must clearly state what the methodology does and does not measure. Specifically: it measures semantic content coherence, not voice/tone consistency, not visual coherence, not consumer perception.

### Intellectual property

- The methodology builds on published academic work (RSA, representation engineering). Not patentable in its general form.
- The specific application to brand coherence measurement, the calibration dataset, and the contrastive fine-tuning approach may be novel enough for a patent, but enforcement would be difficult.
- Stronger IP position: build proprietary calibration data (real-world labeled coherence examples) over time. Each engagement generates labeled data that improves the methodology. This creates a compounding moat.

## 7. Domain Recommendations

### R1: Reframe "coherence" as "semantic content alignment"

The word "coherence" is overloaded in brand management. It evokes voice, tone, visual identity, and emotional positioning — dimensions this methodology does not measure. Recommend framing as "Semantic Content Alignment" or "Message Consistency Index" to set accurate expectations. In client-facing materials, explicitly state: "This methodology measures whether the same product story is being told across channels, independent of how it is told. It does not measure brand voice consistency."

This is especially important given Decision #3 (contrastive fine-tuning to collapse register). The methodology is *intentionally* register-blind. That is a feature for measuring content coherence but a limitation for measuring brand voice. Be precise.

### R2: Design Experiment 1 as a continuum, not a binary

Instead of 10 "consistent" vs. 10 "inconsistent" products, select 20 products spanning a range of expected coherence. Have 3 industry professionals independently rank them from most to least coherent (without seeing the metric). Measure rank-order correlation between the metric and the expert consensus. This provides a more informative validation (Spearman rho rather than a binary classification accuracy) and avoids the confirmation bias risk of hand-picking the two groups.

If the 10-vs-10 binary design is retained, add a blinding step: have the products assigned to groups by one person and the coherence scores interpreted by a different person.

### R3: Separate the two-tier model in the experimental validation

Decision #7 (brand coherence vs. market coherence) is excellent domain thinking. Make this separation explicit in the experiments:
- "Brand Coherence Score" = computed only on brand-controlled and platform-mediated channels.
- "Market Coherence Score" = includes third-party content (reviews, media, social mentions).
- The relationship between the two scores is itself a diagnostic: high brand coherence + low market coherence = "your messaging is disciplined but the market isn't hearing it." Low brand coherence + high market coherence = "consumers have a clearer picture of your product than you do."

### R4: Build the competitive benchmark into the standard deliverable

Experiment 5 is positioned as the last validation step, but competitive context should be part of every commercial engagement. Even if the competitive ranking isn't "validated" by Experiment 5 at the time of first sale, providing it as "indicative" adds enormous value. Brand managers think comparatively. "Your score is 0.72" means nothing. "Your score is 0.72 and your top competitor is 0.81" is actionable.

### R5: Pilot with the rebranding use case

The rebranding edge case (measuring rollout completeness across channels) is the most compelling initial use case because:
- The client already knows messaging is inconsistent (they just changed it).
- The value proposition is clear: track which channels have adopted the new messaging.
- Success criteria are unambiguous: the metric should show increasing coherence over time as more channels are updated.
- It does not require the methodology to detect subtle inconsistencies — just binary old-vs-new messaging detection.

This use case could serve as a "proof of concept" engagement at a lower price point ($10-15K) before offering full portfolio audits.

### R6: Address the voice/tone gap explicitly

Since the contrastive fine-tuning (Decision #3) deliberately collapses register, the methodology cannot detect voice/tone inconsistencies. This is the #1 thing brand managers mean by "coherence." Three options:
1. **Scope limitation:** Accept that this methodology measures content coherence only. Market it as such. Pair it with a separate voice/tone analysis (could be as simple as a style classifier) for a complete offering.
2. **Dual-mode analysis:** Run the pipeline twice — once with the contrastive model (content coherence) and once with the base model (voice/tone coherence). Compare results. A product with high content coherence but low voice coherence has the right message in the wrong style.
3. **Register similarity as a separate metric:** Compute register similarity using the base model (before fine-tuning) as a "Voice Consistency Score." This directly leverages the finding that register is easily classifiable. A brand whose TikTok content has the same register signature as its website content has high voice consistency.

Option 3 is most elegant and leverages the original experiment's findings rather than treating them as a problem to solve.

### R7: Plan for data compounding

Each engagement generates labeled, real-world data: real documents, channel labels, expert coherence judgments, and (eventually) client feedback on whether recommendations were acted upon and effective. This data has three uses:
1. **Calibration improvement:** Better reference distributions for the coherence score as you accumulate more products.
2. **Supervised fine-tuning:** Enough labeled examples could train a supervised coherence predictor, bypassing the RSA-based approach entirely. This is the long-term moat.
3. **Benchmark publication:** Anonymized, aggregated data on cross-channel coherence norms by category becomes a thought-leadership asset. "CPG brands average 0.67 coherence; pharma OTC averages 0.79; DTC brands average 0.54."

### R8: Validate that the commercial framing survives first contact with a buyer

Before investing in all five experiments, have 2-3 conversations with potential buyers (brand directors, agency strategists). Show them a mock report for a hypothetical product. Ask: "Would you pay $50K for this? What would make it more valuable? What's missing?" The answers will reshape the experiment priorities. If every buyer says "I need voice consistency, not content consistency," that changes the technical approach fundamentally. If every buyer says "I need this for rebranding tracking," that changes the experiment sequence.

These conversations cost zero and could prevent months of technically sound but commercially misaligned work.
