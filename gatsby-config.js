module.exports = {
  siteMetadata: {
    title: `ATech Guide`,
    description: `Kamran Ali's Tech Blog`,
    author: `Kamran Ali`,
    twitterId: `@aTechGuide`,
    //siteUrl: `https://atech.guide`,
    siteUrl: `https://atech-guide.netlify.com`,
    genre: 'Technical Tutorials',
    keywords: [`Technology Blog`],
    email: `admin@atech.guide`,
    social: [
      'https://www.facebook.com/aTechGuide/'
    ],
    contactSupport: 'https://www.facebook.com/aTechGuide/',
    bingId: '',
    menuLinks: [{name: 'Tags', link: '/tags/'}],
    footerLinks: [{name: 'About', link: '/detailed-tech-tutorials/'}, {name: 'Terms of Use', link: '/terms-of-use/'}, {name: 'Privacy Policy', link: '/privacy-policy/'}],
    displayFooterMessage: true,
    comments: 'true' // Enable disable comments
  },
  //__experimentalThemes: ['gatsby-theme-blog-starter'],
  plugins: [
    {
      resolve: "gatsby-theme-blog-starter",
      options: {
        trackingId: "UA-27634418-4",
        postsPath: "posts",
        postsPerPage: "12",
        mailchimpURL: "https://kamranali.us17.list-manage.com/subscribe/post?u=835b966c8e4fb4811d20a1b0c&amp;id=1ccb85525c"
      },
    },
    {
      resolve: `gatsby-plugin-manifest`, //<- Creates manifest file
      options: {
        name: "ATechGuide",
        short_name: "ATechGuide",
        description: "Tech Blog",
        start_url: "/",
        background_color: "#673ab7",
        theme_color: "#673ab7",
        display: "standalone",
        icon: "images/icon.png",
      },
    },
    `gatsby-plugin-offline`, //<- Adds service worker; Add after gatsby-plugin-manifest
    {
      resolve: 'gatsby-plugin-netlify',
      options: {
        headers: {
          '/*': [
            'X-Frame-Options: DENY',
            'X-XSS-Protection: 1; mode=block',
            'X-Content-Type-Options: nosniff',
            'Referrer-Policy: no-referrer-when-downgrade'
          ]
        }
      }
    },
    {
      resolve: `gatsby-plugin-csp`,
      options: {
        disableOnDev: true,
        mergeScriptHashes: false,
        mergeStyleHashes: false, 
        mergeDefaultDirectives: true,
        directives: {
          "default-src": "'self' https://disqus.com https://c.disquscdn.com www.google-analytics.com www.google-analytics.com/analytics.js",
          "script-src": "'self' 'unsafe-inline' www.google-analytics.com https://kamranali.disqus.com", //<- 'unsafe-inline' is unsafe and is required by Disqus
          "style-src": "'self' 'unsafe-inline' https://c.disquscdn.com", //<- "'unsafe-inline'" should be avoided but the plugin was broken with mergeStyleHashes
          "img-src": "'self' data: www.google-analytics.com https://referrer.disqus.com https://c.disquscdn.com"
        }
      }
    }
  ]
}
