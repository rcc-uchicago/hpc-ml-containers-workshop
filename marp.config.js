module.exports = {
  // Core configurations

  // Customize engine through markdown-it plugins
  engine: ({ marp }) => {
    const kroki = require('@kazumatu981/markdown-it-kroki');
    
    // Configure Kroki integration - use the public Kroki instance
    marp.use(kroki, {
      // Kroki server URL - using public instance
      server: 'https://kroki.io',
      // Default diagram type if not specified
      defaultDiagramType: 'mermaid'
    });
    
    return marp;
  },
  
  // Additional configurations
  html: true,
  allowLocalFiles: true,
  themeSet: './uchicago-rcc.css'
};
