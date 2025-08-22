// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AIKit',
  tagline: 'Fine-tune, build, and deploy open-source LLMs easily!',
  favicon: 'img/favicon.ico',
  headTags: [
    {
      tagName: "meta",
      attributes: {
        // Allow Algolia crawler to index the site
        // See https://www.algolia.com/doc/tools/crawler/getting-started/create-crawler/#verify-your-domain.
        name: "algolia-site-verification",
        content: "58101301D914B63C",
      }
    },
  ],

  // Set the production url of your site here
  url: 'https://kaito-project.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/aikit/docs/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'kaito-project', // Usually your GitHub org/user name.
  projectName: 'aikit', // Usually your repo name.

  onBrokenLinks: 'throw', // throw
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/kaito-project/aikit/blob/main/website/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/logo.png',
      navbar: {
        title: 'AIKit',
        logo: {
          alt: 'AIKit Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            href: 'https://github.com/kaito-project/aikit',
            position: 'right',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} Sertac Ozercan`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'json', 'yaml'],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      announcementBar: {
        id: 'announcementBar-1', // Increment on change
        content: `⭐️ If you like AIKit, please give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/kaito-project/aikit">GitHub</a>!</a>`,
      },
      algolia: {
        appId: 'BWYV6PMJ5K',
        apiKey: 'e2cfa004b0a812062660e0039aca0bda',
        indexName: 'aikit-crawler',
      },
    }),
};

export default config;
