const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  await page.goto('http://localhost:8000/web-concept-clustering/index.html');
  await page.waitForSelector('#process-btn');
  await page.click('#process-btn');
  await new Promise(r => setTimeout(r, 5000));
  await browser.close();
})();
