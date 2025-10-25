/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  sidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'intro',
        'quick-start',
        'premade-models',
        'demo',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      collapsed: false,
      items: [
        'create-images',
        'fine-tune',
        'packaging',
        'vision',
        'gpu',
        'kubernetes',
        'security',
      ],
    },
    {
      type: 'category',
      label: 'Specifications',
      collapsed: false,
      items: [
        'specs-inference',
        'specs-finetune',
      ],
    },
    {
      type: 'category',
      label: 'Inference Supported Backends',
      collapsed: false,
      items: [
        'llama-cpp',
        'exllama2',
        'diffusion',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      collapsed: false,
      items: [
        'architecture',
        'release',
      ],
    },
  ],
};

export default sidebars;
