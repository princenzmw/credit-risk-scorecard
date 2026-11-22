/**
 * Cloudflare Worker: Credio Crawler OG Prerender & Transparent Proxy
 *
 * Intercepts social media crawlers and bots requesting credio.nzamuwe.com/*
 * and serves complete OpenGraph & Twitter card HTML metadata.
 * Forwards all human users and browser sessions directly to Northflank Streamlit origin.
 */

const BOT_PATTERNS = [
  /bot/i,
  /spider/i,
  /crawl/i,
  /facebookexternalhit/i,
  /twitterbot/i,
  /linkedinbot/i,
  /slackbot/i,
  /whatsapp/i,
  /telegrambot/i,
  /discordbot/i,
  /applebot/i,
  /pinterest/i,
  /googlebot/i,
  /bingbot/i,
];

function isCrawler(userAgent) {
  if (!userAgent) return false;
  return BOT_PATTERNS.some((pattern) => pattern.test(userAgent));
}

function renderCrawlerHtml() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Credio: Credit-Risk Decisioning</title>
  <meta name="description" content="Predicting borrower default risk with 0.869 ROC-AUC on 150,000 credit records using a cost-calibrated gradient boosting pipeline.">
  <link rel="canonical" href="https://credio.nzamuwe.com/">

  <!-- OpenGraph Metadata -->
  <meta property="og:type" content="website">
  <meta property="og:title" content="Credio: Credit-Risk Decisioning">
  <meta property="og:description" content="Predicting borrower default risk with 0.869 ROC-AUC on 150,000 credit records using a cost-calibrated gradient boosting pipeline.">
  <meta property="og:url" content="https://credio.nzamuwe.com/">
  <meta property="og:image" content="https://nzamuwe.com/assets/credio/og-image.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="Credio AI-Assisted Credit Decisioning Scorecard">

  <!-- Twitter Card Metadata -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Credio: Credit-Risk Decisioning">
  <meta name="twitter:description" content="Predicting borrower default risk with 0.869 ROC-AUC on 150,000 credit records using a cost-calibrated gradient boosting pipeline.">
  <meta name="twitter:image" content="https://nzamuwe.com/assets/credio/og-image.png">
</head>
<body>
  <main style="font-family: system-ui, sans-serif; padding: 40px; max-width: 600px; margin: auto;">
    <h1>Credio: Credit-Risk Decisioning</h1>
    <p>Predicting borrower default risk with 0.869 ROC-AUC on 150,000 credit records using a cost-calibrated gradient boosting pipeline.</p>
    <p><a href="https://credio.nzamuwe.com/">Open Interactive Streamlit Application &rarr;</a></p>
  </main>
</body>
</html>`;
}

export default {
  async fetch(request, env, ctx) {
    const userAgent = request.headers.get("user-agent") || "";

    if (isCrawler(userAgent)) {
      return new Response(renderCrawlerHtml(), {
        status: 200,
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "public, max-age=86400",
        },
      });
    }

    // Pass through directly to origin (Northflank)
    return fetch(request);
  },
};
