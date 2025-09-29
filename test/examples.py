import time

documents = [
    {
        "metadata": {
            "doc_id": "doc_001_research_paper",
            "filename": "ai_transformers_overview.pdf",
            "filepath": "/documents/research/ai_transformers_overview.pdf",
            "file_size": 2048576,  # ~2MB
            "last_modified": time.time(),
            "created": time.time() - 86400,  # Created 1 day ago
            "extension": "pdf",
            "mime_type": "application/pdf",
            "language": "en",
            "full_char_count": 15420
        },
        "full_text": """
        Transformer Architecture in Modern AI Systems
        
        Abstract: This paper provides a comprehensive overview of transformer architectures 
        and their applications in natural language processing. The transformer model, 
        introduced by Vaswani et al. in 2017, has revolutionized the field of machine 
        learning through its attention mechanism and parallel processing capabilities.
        
        Introduction: Traditional recurrent neural networks process sequences sequentially, 
        which limits their ability to capture long-range dependencies efficiently. The 
        transformer architecture addresses these limitations through self-attention 
        mechanisms that allow the model to focus on relevant parts of the input sequence 
        regardless of their position.
        
        Key Contributions:
        - Self-attention mechanism for improved context understanding
        - Parallel processing capabilities for faster training
        - Positional encoding for sequence order preservation
        - Multi-head attention for diverse representation learning
        
        Conclusion: Transformers have become the foundation for state-of-the-art models 
        in natural language processing, computer vision, and other domains, demonstrating 
        remarkable versatility and performance improvements over previous architectures.
        """
    },
    
    {
        "metadata": {
            "doc_id": "doc_002_meeting_notes",
            "filename": "weekly_team_meeting_2025_01_15.txt",
            "filepath": "/documents/meetings/weekly_team_meeting_2025_01_15.txt",
            "file_size": 4096,  # ~4KB
            "last_modified": time.time() - 3600,  # Modified 1 hour ago
            "created": time.time() - 7200,  # Created 2 hours ago
            "extension": "txt",
            "mime_type": "text/plain",
            "language": "en",
            "full_char_count": 2847
        },
        "full_text": """
        Weekly Team Meeting - January 15, 2025
        
        Attendees:
        - Sarah Johnson (Product Manager)
        - Mike Chen (Lead Developer)
        - Emily Rodriguez (UX Designer)
        - David Kim (Data Scientist)
        - Alex Thompson (QA Engineer)
        
        Agenda Items:
        
        1. Sprint Review
        - Completed 8 out of 10 planned user stories
        - Performance improvements implemented successfully
        - Two stories moved to next sprint due to scope changes
        
        2. Current Blockers
        - Database migration delayed pending infrastructure approval
        - Third-party API integration needs security review
        - UI mockups awaiting stakeholder feedback
        
        3. Upcoming Priorities
        - Focus on user authentication improvements
        - Prepare for Q1 security audit
        - Begin planning for mobile app features
        
        Action Items:
        - Sarah: Follow up with infrastructure team by Friday
        - Mike: Complete API documentation by Wednesday
        - Emily: Schedule stakeholder review meeting
        - David: Prepare performance metrics report
        - Alex: Update test automation framework
        
        Next Meeting: January 22, 2025 at 10:00 AM
        """
    },
    
    {
        "metadata": {
            "doc_id": "doc_003_recipe_collection",
            "filename": "grandmas_recipes.md",
            "filepath": "/documents/personal/grandmas_recipes.md",
            "file_size": 8192,  # ~8KB
            "last_modified": time.time() - 86400 * 7,  # Modified 1 week ago
            "created": time.time() - 86400 * 30,  # Created 1 month ago
            "extension": "md",
            "mime_type": "text/markdown",
            "language": "en",
            "full_char_count": 5623
        },
        "full_text": """
        # Grandma's Recipe Collection
        
        ## Classic Chocolate Chip Cookies
        
        ### Ingredients:
        - 2¼ cups all-purpose flour
        - 1 tsp baking soda
        - 1 tsp salt
        - 1 cup butter, softened
        - ¾ cup granulated sugar
        - ¾ cup brown sugar, packed
        - 2 large eggs
        - 2 tsp vanilla extract
        - 2 cups chocolate chips
        
        ### Instructions:
        1. Preheat oven to 375°F (190°C)
        2. Mix flour, baking soda, and salt in a bowl
        3. Beat butter and sugars until creamy
        4. Add eggs and vanilla, mix well
        5. Gradually add flour mixture
        6. Stir in chocolate chips
        7. Drop rounded tablespoons onto ungreased baking sheets
        8. Bake 9-11 minutes until golden brown
        9. Cool on baking sheet for 2 minutes before removing
        
        ## Homemade Chicken Soup
        
        ### Ingredients:
        - 1 whole chicken (3-4 lbs)
        - 8 cups water
        - 2 carrots, sliced
        - 2 celery stalks, chopped
        - 1 onion, diced
        - 2 bay leaves
        - Salt and pepper to taste
        - 1 cup egg noodles
        - Fresh parsley for garnish
        
        ### Instructions:
        1. Place chicken in large pot with water
        2. Bring to boil, then simmer for 1 hour
        3. Remove chicken, shred meat, discard bones
        4. Add vegetables and bay leaves to broth
        5. Simmer 20 minutes until vegetables are tender
        6. Add noodles and cook 8-10 minutes
        7. Return shredded chicken to pot
        8. Season with salt, pepper, and parsley
        9. Serve hot with crackers
        
        ## Notes:
        These recipes have been passed down through three generations. 
        The secret to the cookies is using room temperature butter and 
        not overbaking them. For the soup, homemade broth makes all 
        the difference!
        """
    }
]

chunks = [
    # Document 1: AI Transformers Research Paper - 5 chunks
    {
        "chunk_uuid": "ce3e73a2-7d1a-4111-b374-c4388c0b966d",
        "doc_id": "doc_001_research_paper",
        "chunk_id": 1,
        "filename": "ai_transformers_overview.pdf",
        "text": "Transformer Architecture in Modern AI Systems\n\nAbstract: This paper provides a comprehensive overview of transformer architectures and their applications in natural language processing. The transformer model, introduced by Vaswani et al. in 2017, has revolutionized the field of machine learning through its attention mechanism and parallel processing capabilities.",
        "char_count": 395,
        "start_page": 1,
        "end_page": 1
    },
    {
        "chunk_uuid": '91c96a29-dc84-45a8-8e09-15645c8a8c02',
        "doc_id": "doc_001_research_paper",
        "chunk_id": 2,
        "filename": "ai_transformers_overview.pdf",
        "text": "Introduction: Traditional recurrent neural networks process sequences sequentially, which limits their ability to capture long-range dependencies efficiently. The transformer architecture addresses these limitations through self-attention mechanisms that allow the model to focus on relevant parts of the input sequence regardless of their position.",
        "char_count": 350,
        "start_page": 1,
        "end_page": 2
    },
    {
        "chunk_uuid": 'bb95f0a9-bc63-4962-bce6-e9966c7d2c92',
        "doc_id": "doc_001_research_paper",
        "chunk_id": 3,
        "filename": "ai_transformers_overview.pdf",
        "text": "Key Contributions:\n- Self-attention mechanism for improved context understanding\n- Parallel processing capabilities for faster training\n- Positional encoding for sequence order preservation\n- Multi-head attention for diverse representation learning",
        "char_count": 252,
        "start_page": 2,
        "end_page": 2
    },
    {
        "chunk_uuid": '4ae9ba1d-ea8b-4d1c-b768-425d767576de',
        "doc_id": "doc_001_research_paper",
        "chunk_id": 4,
        "filename": "ai_transformers_overview.pdf",
        "text": "Conclusion: Transformers have become the foundation for state-of-the-art models in natural language processing, computer vision, and other domains, demonstrating remarkable versatility and performance improvements over previous architectures.",
        "char_count": 226,
        "start_page": 3,
        "end_page": 3
    },
    {
        "chunk_uuid": 'a8ce29a9-1f26-435d-8d1f-dfd96368af8d',
        "doc_id": "doc_001_research_paper",
        "chunk_id": 5,
        "filename": "ai_transformers_overview.pdf",
        "text": "The transformer model has revolutionized machine learning through attention mechanisms and parallel processing capabilities, addressing limitations of traditional recurrent neural networks in capturing long-range dependencies efficiently.",
        "char_count": 215,
        "start_page": 1,
        "end_page": 3
    },

    # Document 2: Weekly Team Meeting Notes - 5 chunks
    {
        "chunk_uuid": 'b95c346b-1802-41cd-96de-d8a5826fa200',
        "doc_id": "doc_002_meeting_notes",
        "chunk_id": 1,
        "filename": "weekly_team_meeting_2025_01_15.txt",
        "text": "Weekly Team Meeting - January 15, 2025\n\nAttendees:\n- Sarah Johnson (Product Manager)\n- Mike Chen (Lead Developer)\n- Emily Rodriguez (UX Designer)\n- David Kim (Data Scientist)\n- Alex Thompson (QA Engineer)",
        "char_count": 208,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '750a0cb7-d488-4226-a45c-9a56811162d7',
        "doc_id": "doc_002_meeting_notes",
        "chunk_id": 2,
        "filename": "weekly_team_meeting_2025_01_15.txt",
        "text": "1. Sprint Review\n- Completed 8 out of 10 planned user stories\n- Performance improvements implemented successfully\n- Two stories moved to next sprint due to scope changes",
        "char_count": 157,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": 'f3b5c5b4-5c5e-4c5b-8c5b-5c5b5c5b5c5b',
        "doc_id": "doc_002_meeting_notes",
        "chunk_id": 3,
        "filename": "weekly_team_meeting_2025_01_15.txt",
        "text": "2. Current Blockers\n- Database migration delayed pending infrastructure approval\n- Third-party API integration needs security review\n- UI mockups awaiting stakeholder feedback",
        "char_count": 166,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '3c9f5c5b-5c5e-4c5b-8c5b-5c5b5c5b5c5b',
        "doc_id": "doc_002_meeting_notes",
        "chunk_id": 4,
        "filename": "weekly_team_meeting_2025_01_15.txt",
        "text": "3. Upcoming Priorities\n- Focus on user authentication improvements\n- Prepare for Q1 security audit\n- Begin planning for mobile app features",
        "char_count": 134,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '46d45855-4286-48b9-93e2-b538905f0222',
        "doc_id": "doc_002_meeting_notes",
        "chunk_id": 5,
        "filename": "weekly_team_meeting_2025_01_15.txt",
        "text": "Action Items:\n- Sarah: Follow up with infrastructure team by Friday\n- Mike: Complete API documentation by Wednesday\n- Emily: Schedule stakeholder review meeting\n- David: Prepare performance metrics report\n- Alex: Update test automation framework\n\nNext Meeting: January 22, 2025 at 10:00 AM",
        "char_count": 313,
        "start_page": None,
        "end_page": None
    },

    # Document 3: Grandma's Recipe Collection - 5 chunks
    {
        "chunk_uuid": 'dce983d4-888e-4c83-a8d0-fadd7404058b',
        "doc_id": "doc_003_recipe_collection",
        "chunk_id": 1,
        "filename": "grandmas_recipes.md",
        "text": "# Grandma's Recipe Collection\n\n## Classic Chocolate Chip Cookies\n\n### Ingredients:\n- 2¼ cups all-purpose flour\n- 1 tsp baking soda\n- 1 tsp salt\n- 1 cup butter, softened\n- ¾ cup granulated sugar\n- ¾ cup brown sugar, packed\n- 2 large eggs\n- 2 tsp vanilla extract\n- 2 cups chocolate chips",
        "char_count": 304,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '30b5998f-64e0-4dc4-a995-ba49279ba2a8',
        "doc_id": "doc_003_recipe_collection",
        "chunk_id": 2,
        "filename": "grandmas_recipes.md",
        "text": "### Instructions:\n1. Preheat oven to 375°F (190°C)\n2. Mix flour, baking soda, and salt in a bowl\n3. Beat butter and sugars until creamy\n4. Add eggs and vanilla, mix well\n5. Gradually add flour mixture\n6. Stir in chocolate chips\n7. Drop rounded tablespoons onto ungreased baking sheets\n8. Bake 9-11 minutes until golden brown\n9. Cool on baking sheet for 2 minutes before removing",
        "char_count": 414,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '83655f0c-c6f9-4572-aec3-8a4103d67199',
        "doc_id": "doc_003_recipe_collection",
        "chunk_id": 3,
        "filename": "grandmas_recipes.md",
        "text": "## Homemade Chicken Soup\n\n### Ingredients:\n- 1 whole chicken (3-4 lbs)\n- 8 cups water\n- 2 carrots, sliced\n- 2 celery stalks, chopped\n- 1 onion, diced\n- 2 bay leaves\n- Salt and pepper to taste\n- 1 cup egg noodles\n- Fresh parsley for garnish",
        "char_count": 252,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '70949b44-8d6c-4171-9590-0f7d58f60ea4',
        "doc_id": "doc_003_recipe_collection",
        "chunk_id": 4,
        "filename": "grandmas_recipes.md",
        "text": "### Instructions:\n1. Place chicken in large pot with water\n2. Bring to boil, then simmer for 1 hour\n3. Remove chicken, shred meat, discard bones\n4. Add vegetables and bay leaves to broth\n5. Simmer 20 minutes until vegetables are tender\n6. Add noodles and cook 8-10 minutes\n7. Return shredded chicken to pot\n8. Season with salt, pepper, and parsley\n9. Serve hot with crackers",
        "char_count": 413,
        "start_page": None,
        "end_page": None
    },
    {
        "chunk_uuid": '8de720dd-3311-475d-9c5b-bcabd28a3c1e',
        "doc_id": "doc_003_recipe_collection",
        "chunk_id": 5,
        "filename": "grandmas_recipes.md",
        "text": "## Notes:\nThese recipes have been passed down through three generations. The secret to the cookies is using room temperature butter and not overbaking them. For the soup, homemade broth makes all the difference!",
        "char_count": 205,
        "start_page": None,
        "end_page": None
    }
]

nodes = [
    # Document 1: AI Transformers Research Paper - Chunk 1
    {
        "name": "Transformer Architecture",
        "type": "CONCEPT",
        "description": "Revolutionary AI model architecture introduced by Vaswani et al. in 2017 with attention mechanism and parallel processing capabilities",
        "source_id": "ce3e73a2-7d1a-4111-b374-c4388c0b966d",
        "filepath": "ai_transformers_overview.pdf"
    },
    {
        "name": "Vaswani et al.",
        "type": "AUTHOR",
        "description": "Research team that introduced the transformer model in 2017, revolutionizing machine learning",
        "source_id": "ce3e73a2-7d1a-4111-b374-c4388c0b966d",
        "filepath": "ai_transformers_overview.pdf"
    },

    # Document 1: AI Transformers Research Paper - Chunk 2
    {
        "name": "Self-Attention Mechanism",
        "type": "TECHNIQUE",
        "description": "Core component of transformer architecture that allows models to focus on relevant parts of input sequence regardless of position",
        "source_id": "91c96a29-dc84-45a8-8e09-15645c8a8c02",
        "filepath": "ai_transformers_overview.pdf"
    },
    {
        "name": "Recurrent Neural Networks",
        "type": "CONCEPT",
        "description": "Traditional neural network architecture that processes sequences sequentially, limiting long-range dependency capture",
        "source_id": "91c96a29-dc84-45a8-8e09-15645c8a8c02",
        "filepath": "ai_transformers_overview.pdf"
    },

    # Document 1: AI Transformers Research Paper - Chunk 3
    {
        "name": "Multi-Head Attention",
        "type": "TECHNIQUE",
        "description": "Transformer component that enables diverse representation learning through multiple attention mechanisms",
        "source_id": "bb95f0a9-bc63-4962-bce6-e9966c7d2c92",
        "filepath": "ai_transformers_overview.pdf"
    },
    {
        "name": "Positional Encoding",
        "type": "TECHNIQUE",
        "description": "Method used in transformers to preserve sequence order information during parallel processing",
        "source_id": "bb95f0a9-bc63-4962-bce6-e9966c7d2c92",
        "filepath": "ai_transformers_overview.pdf"
    },

    # Document 1: AI Transformers Research Paper - Chunk 4
    {
        "name": "Natural Language Processing",
        "type": "DOMAIN",
        "description": "Field of AI where transformers have become the foundation for state-of-the-art models",
        "source_id": "4ae9ba1d-ea8b-4d1c-b768-425d767576de",
        "filepath": "ai_transformers_overview.pdf"
    },
    {
        "name": "Computer Vision",
        "type": "DOMAIN",
        "description": "AI domain where transformers demonstrate remarkable versatility and performance improvements",
        "source_id": "4ae9ba1d-ea8b-4d1c-b768-425d767576de",
        "filepath": "ai_transformers_overview.pdf"
    },

    # Document 1: AI Transformers Research Paper - Chunk 5
    {
        "name": "Machine Learning Revolution",
        "type": "CONCEPT",
        "description": "Transformation in ML field driven by transformer models through attention mechanisms and parallel processing",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf"
    },
    {
        "name": "Long-Range Dependencies",
        "type": "CONCEPT",
        "description": "Relationships between distant elements in sequences that transformers can capture more efficiently than RNNs",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf"
    },

    # Document 2: Weekly Team Meeting Notes - Chunk 1
    {
        "name": "Sarah Johnson",
        "type": "PERSON",
        "description": "Product Manager attending the weekly team meeting on January 15, 2025",
        "source_id": "b95c346b-1802-41cd-96de-d8a5826fa200",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },
    {
        "name": "Development Team",
        "type": "ORGANIZATION",
        "description": "Cross-functional team including Product Manager, Developer, UX Designer, Data Scientist, and QA Engineer",
        "source_id": "b95c346b-1802-41cd-96de-d8a5826fa200",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },

    # Document 2: Weekly Team Meeting Notes - Chunk 2
    {
        "name": "Sprint Review",
        "type": "PROCESS",
        "description": "Team completed 8 out of 10 planned user stories with successful performance improvements implementation",
        "source_id": "750a0cb7-d488-4226-a45c-9a56811162d7",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },
    {
        "name": "Performance Improvements",
        "type": "DELIVERABLE",
        "description": "Successfully implemented enhancements to system performance during the current sprint",
        "source_id": "750a0cb7-d488-4226-a45c-9a56811162d7",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },

    # Document 2: Weekly Team Meeting Notes - Chunk 3
    {
        "name": "Database Migration",
        "type": "TASK",
        "description": "Critical project blocker delayed pending infrastructure team approval",
        "source_id": "f3b5c5b4-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },
    {
        "name": "Third-Party API Integration",
        "type": "TASK",
        "description": "Development work requiring security review before implementation can proceed",
        "source_id": "f3b5c5b4-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },

    # Document 2: Weekly Team Meeting Notes - Chunk 4
    {
        "name": "User Authentication Improvements",
        "type": "PRIORITY",
        "description": "Upcoming focus area for the development team to enhance system security",
        "source_id": "3c9f5c5b-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },
    {
        "name": "Q1 Security Audit",
        "type": "MILESTONE",
        "description": "Upcoming security assessment that the team needs to prepare for",
        "source_id": "3c9f5c5b-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },

    # Document 2: Weekly Team Meeting Notes - Chunk 5
    {
        "name": "Mike Chen",
        "type": "PERSON",
        "description": "Lead Developer responsible for completing API documentation by Wednesday",
        "source_id": "46d45855-4286-48b9-93e2-b538905f0222",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },
    {
        "name": "API Documentation",
        "type": "DELIVERABLE",
        "description": "Technical documentation task assigned to Mike Chen with Wednesday deadline",
        "source_id": "46d45855-4286-48b9-93e2-b538905f0222",
        "filepath": "weekly_team_meeting_2025_01_15.txt"
    },

    # Document 3: Grandma's Recipe Collection - Chunk 1
    {
        "name": "Chocolate Chip Cookies",
        "type": "RECIPE",
        "description": "Classic cookie recipe requiring all-purpose flour, butter, sugars, eggs, vanilla, and chocolate chips",
        "source_id": "dce983d4-888e-4c83-a8d0-fadd7404058b",
        "filepath": "grandmas_recipes.md"
    },
    {
        "name": "Baking Ingredients",
        "type": "CATEGORY",
        "description": "Essential ingredients for cookie making including flour, baking soda, butter, sugars, and chocolate chips",
        "source_id": "dce983d4-888e-4c83-a8d0-fadd7404058b",
        "filepath": "grandmas_recipes.md"
    },

    # Document 3: Grandma's Recipe Collection - Chunk 2
    {
        "name": "Cookie Baking Process",
        "type": "PROCEDURE",
        "description": "Step-by-step instructions for making cookies from mixing ingredients to baking at 375°F for 9-11 minutes",
        "source_id": "30b5998f-64e0-4dc4-a995-ba49279ba2a8",
        "filepath": "grandmas_recipes.md"
    },
    {
        "name": "Oven Temperature",
        "type": "PARAMETER",
        "description": "Specific baking temperature of 375°F (190°C) required for proper cookie preparation",
        "source_id": "30b5998f-64e0-4dc4-a995-ba49279ba2a8",
        "filepath": "grandmas_recipes.md"
    },

    # Document 3: Grandma's Recipe Collection - Chunk 3
    {
        "name": "Chicken Soup",
        "type": "RECIPE",
        "description": "Homemade soup recipe using whole chicken, vegetables, and egg noodles in a hearty broth",
        "source_id": "83655f0c-c6f9-4572-aec3-8a4103d67199",
        "filepath": "grandmas_recipes.md"
    },
    {
        "name": "Soup Vegetables",
        "type": "INGREDIENT_GROUP",
        "description": "Fresh vegetables for soup including carrots, celery, onion, and parsley for flavor and nutrition",
        "source_id": "83655f0c-c6f9-4572-aec3-8a4103d67199",
        "filepath": "grandmas_recipes.md"
    },

    # Document 3: Grandma's Recipe Collection - Chunk 4
    {
        "name": "Soup Preparation Method",
        "type": "PROCEDURE",
        "description": "Complete process from simmering whole chicken to adding vegetables, noodles, and seasoning",
        "source_id": "70949b44-8d6c-4171-9590-0f7d58f60ea4",
        "filepath": "grandmas_recipes.md"
    },
    {
        "name": "Chicken Broth",
        "type": "BASE_INGREDIENT",
        "description": "Homemade broth created by simmering whole chicken for 1 hour, forming the soup's foundation",
        "source_id": "70949b44-8d6c-4171-9590-0f7d58f60ea4",
        "filepath": "grandmas_recipes.md"
    },

    # Document 3: Grandma's Recipe Collection - Chunk 5
    {
        "name": "Family Recipe Tradition",
        "type": "CONCEPT",
        "description": "Three-generation tradition of passing down cooking knowledge and recipe secrets",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md"
    },
    {
        "name": "Cooking Secrets",
        "type": "KNOWLEDGE",
        "description": "Traditional cooking tips including using room temperature butter for cookies and homemade broth for soup",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md"
    }
]

edges = [
    # AI Transformers Research Paper - Internal relationships
    {
        "source_name": "Transformer Architecture",
        "target_name": "Vaswani et al.",
        "weight": 0.9,
        "description": "Transformer architecture was introduced by Vaswani et al. in 2017",
        "keywords": "introduction, research, authorship",
        "source_id": "ce3e73a2-7d1a-4111-b374-c4388c0b966d",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Transformer Architecture",
        "u_target_name": "Vaswani et al."
    },
    {
        "source_name": "Self-Attention Mechanism",
        "target_name": "Transformer Architecture",
        "weight": 0.95,
        "description": "Self-attention mechanism is the core component of transformer architecture",
        "keywords": "component, core, architecture",
        "source_id": "91c96a29-dc84-45a8-8e09-15645c8a8c02",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Self-Attention Mechanism",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Multi-Head Attention",
        "target_name": "Transformer Architecture",
        "weight": 0.9,
        "description": "Multi-head attention is a key component of transformer architecture",
        "keywords": "component, attention, multiple",
        "source_id": "bb95f0a9-bc63-4962-bce6-e9966c7d2c92",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Multi-Head Attention",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Multi-Head Attention",
        "target_name": "Self-Attention Mechanism",
        "weight": 0.85,
        "description": "Multi-head attention extends self-attention mechanism with multiple attention heads",
        "keywords": "extension, multiple, attention",
        "source_id": "bb95f0a9-bc63-4962-bce6-e9966c7d2c92",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Multi-Head Attention",
        "u_target_name": "Self-Attention Mechanism"
    },
    {
        "source_name": "Positional Encoding",
        "target_name": "Transformer Architecture",
        "weight": 0.85,
        "description": "Positional encoding preserves sequence order in transformer architecture",
        "keywords": "sequence, order, position",
        "source_id": "bb95f0a9-bc63-4962-bce6-e9966c7d2c92",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Positional Encoding",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Recurrent Neural Networks",
        "target_name": "Transformer Architecture",
        "weight": 0.7,
        "description": "RNNs are contrasted with transformers as traditional sequential processing approach",
        "keywords": "contrast, traditional, sequential",
        "source_id": "91c96a29-dc84-45a8-8e09-15645c8a8c02",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Recurrent Neural Networks",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Long-Range Dependencies",
        "target_name": "Recurrent Neural Networks",
        "weight": 0.8,
        "description": "RNNs have limitations in capturing long-range dependencies",
        "keywords": "limitation, capture, sequential",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Long-Range Dependencies",
        "u_target_name": "Recurrent Neural Networks"
    },
    {
        "source_name": "Long-Range Dependencies",
        "target_name": "Self-Attention Mechanism",
        "weight": 0.9,
        "description": "Self-attention mechanism efficiently captures long-range dependencies",
        "keywords": "efficient, capture, attention",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Long-Range Dependencies",
        "u_target_name": "Self-Attention Mechanism"
    },
    {
        "source_name": "Natural Language Processing",
        "target_name": "Transformer Architecture",
        "weight": 0.9,
        "description": "Transformers are foundational for state-of-the-art NLP models",
        "keywords": "foundation, state-of-the-art, application",
        "source_id": "4ae9ba1d-ea8b-4d1c-b768-425d767576de",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Natural Language Processing",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Computer Vision",
        "target_name": "Transformer Architecture",
        "weight": 0.8,
        "description": "Transformers demonstrate versatility and performance in computer vision",
        "keywords": "versatility, performance, application",
        "source_id": "4ae9ba1d-ea8b-4d1c-b768-425d767576de",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Computer Vision",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Machine Learning Revolution",
        "target_name": "Transformer Architecture",
        "weight": 0.95,
        "description": "Transformer architecture drives the machine learning revolution",
        "keywords": "revolution, transformation, paradigm",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Machine Learning Revolution",
        "u_target_name": "Transformer Architecture"
    },
    {
        "source_name": "Machine Learning Revolution",
        "target_name": "Self-Attention Mechanism",
        "weight": 0.85,
        "description": "Self-attention mechanism is a key driver of the ML revolution",
        "keywords": "driver, attention, breakthrough",
        "source_id": "a8ce29a9-1f26-435d-8d1f-dfd96368af8d",
        "filepath": "ai_transformers_overview.pdf",
        "u_source_name": "Machine Learning Revolution",
        "u_target_name": "Self-Attention Mechanism"
    },

    # Weekly Team Meeting - Internal relationships
    {
        "source_name": "Sarah Johnson",
        "target_name": "Development Team",
        "weight": 0.9,
        "description": "Sarah Johnson is the Product Manager in the development team",
        "keywords": "product manager, team member, leadership",
        "source_id": "b95c346b-1802-41cd-96de-d8a5826fa200",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Development Team",
        "u_target_name": "Sarah Johnson"
    },
    {
        "source_name": "Mike Chen",
        "target_name": "Development Team",
        "weight": 0.9,
        "description": "Mike Chen is the Lead Developer in the development team",
        "keywords": "lead developer, team member, technical",
        "source_id": "46d45855-4286-48b9-93e2-b538905f0222",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Development Team",
        "u_target_name": "Mike Chen"
    },
    {
        "source_name": "Sprint Review",
        "target_name": "Development Team",
        "weight": 0.85,
        "description": "Development team conducted sprint review of completed work",
        "keywords": "review, process, completion",
        "source_id": "750a0cb7-d488-4226-a45c-9a56811162d7",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Development Team",
        "u_target_name": "Sprint Review"
    },
    {
        "source_name": "Performance Improvements",
        "target_name": "Sprint Review",
        "weight": 0.9,
        "description": "Performance improvements were successfully implemented during the sprint",
        "keywords": "implementation, success, deliverable",
        "source_id": "750a0cb7-d488-4226-a45c-9a56811162d7",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Performance Improvements",
        "u_target_name": "Sprint Review"
    },
    {
        "source_name": "Database Migration",
        "target_name": "Development Team",
        "weight": 0.8,
        "description": "Database migration is a critical task for the development team",
        "keywords": "critical, blocker, infrastructure",
        "source_id": "f3b5c5b4-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Database Migration",
        "u_target_name": "Development Team"
    },
    {
        "source_name": "Third-Party API Integration",
        "target_name": "Development Team",
        "weight": 0.8,
        "description": "Third-party API integration requires development work and security review",
        "keywords": "integration, security, review",
        "source_id": "f3b5c5b4-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Development Team",
        "u_target_name": "Third-Party API Integration"
    },
    {
        "source_name": "User Authentication Improvements",
        "target_name": "Development Team",
        "weight": 0.85,
        "description": "User authentication improvements are upcoming focus for the development team",
        "keywords": "security, priority, focus",
        "source_id": "3c9f5c5b-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Development Team",
        "u_target_name": "User Authentication Improvements"
    },
    {
        "source_name": "Q1 Security Audit",
        "target_name": "User Authentication Improvements",
        "weight": 0.8,
        "description": "User authentication improvements are needed for Q1 security audit preparation",
        "keywords": "security, audit, preparation",
        "source_id": "3c9f5c5b-5c5e-4c5b-8c5b-5c5b5c5b5c5b",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "Q1 Security Audit",
        "u_target_name": "User Authentication Improvements"
    },
    {
        "source_name": "API Documentation",
        "target_name": "Mike Chen",
        "weight": 0.9,
        "description": "Mike Chen is responsible for completing API documentation by Wednesday",
        "keywords": "responsibility, deadline, documentation",
        "source_id": "46d45855-4286-48b9-93e2-b538905f0222",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "API Documentation",
        "u_target_name": "Mike Chen"
    },
    {
        "source_name": "API Documentation",
        "target_name": "Third-Party API Integration",
        "weight": 0.8,
        "description": "API documentation is related to third-party API integration work",
        "keywords": "documentation, integration, technical",
        "source_id": "46d45855-4286-48b9-93e2-b538905f0222",
        "filepath": "weekly_team_meeting_2025_01_15.txt",
        "u_source_name": "API Documentation",
        "u_target_name": "Third-Party API Integration"
    },

    # Grandma's Recipe Collection - Internal relationships
    {
        "source_name": "Chocolate Chip Cookies",
        "target_name": "Baking Ingredients",
        "weight": 0.9,
        "description": "Chocolate chip cookies recipe requires specific baking ingredients",
        "keywords": "recipe, ingredients, baking",
        "source_id": "dce983d4-888e-4c83-a8d0-fadd7404058b",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Baking Ingredients",
        "u_target_name": "Chocolate Chip Cookies"
    },
    {
        "source_name": "Cookie Baking Process",
        "target_name": "Chocolate Chip Cookies",
        "weight": 0.95,
        "description": "Cookie baking process describes how to make chocolate chip cookies",
        "keywords": "process, procedure, instructions",
        "source_id": "30b5998f-64e0-4dc4-a995-ba49279ba2a8",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chocolate Chip Cookies",
        "u_target_name": "Cookie Baking Process"
    },
    {
        "source_name": "Cookie Baking Process",
        "target_name": "Oven Temperature",
        "weight": 0.85,
        "description": "Cookie baking process requires specific oven temperature of 375°F",
        "keywords": "temperature, baking, requirement",
        "source_id": "30b5998f-64e0-4dc4-a995-ba49279ba2a8",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Cookie Baking Process",
        "u_target_name": "Oven Temperature"
    },
    {
        "source_name": "Chicken Soup",
        "target_name": "Soup Vegetables",
        "weight": 0.9,
        "description": "Chicken soup recipe uses fresh vegetables for flavor and nutrition",
        "keywords": "ingredients, vegetables, nutrition",
        "source_id": "83655f0c-c6f9-4572-aec3-8a4103d67199",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Soup",
        "u_target_name": "Soup Vegetables"
    },
    {
        "source_name": "Soup Preparation Method",
        "target_name": "Chicken Soup",
        "weight": 0.95,
        "description": "Soup preparation method describes how to make chicken soup",
        "keywords": "method, preparation, instructions",
        "source_id": "70949b44-8d6c-4171-9590-0f7d58f60ea4",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Soup",
        "u_target_name": "Soup Preparation Method"
    },
    {
        "source_name": "Chicken Broth",
        "target_name": "Soup Preparation Method",
        "weight": 0.9,
        "description": "Chicken broth is created through the soup preparation method",
        "keywords": "broth, foundation, simmering",
        "source_id": "70949b44-8d6c-4171-9590-0f7d58f60ea4",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Broth",
        "u_target_name": "Soup Preparation Method"
    },
    {
        "source_name": "Chicken Broth",
        "target_name": "Chicken Soup",
        "weight": 0.95,
        "description": "Chicken broth forms the foundation of chicken soup",
        "keywords": "foundation, base, soup",
        "source_id": "70949b44-8d6c-4171-9590-0f7d58f60ea4",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Broth",
        "u_target_name": "Chicken Soup"
    },
    {
        "source_name": "Family Recipe Tradition",
        "target_name": "Chocolate Chip Cookies",
        "weight": 0.8,
        "description": "Chocolate chip cookies are part of three-generation family recipe tradition",
        "keywords": "tradition, family, heritage",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chocolate Chip Cookies",
        "u_target_name": "Family Recipe Tradition"
    },
    {
        "source_name": "Family Recipe Tradition",
        "target_name": "Chicken Soup",
        "weight": 0.8,
        "description": "Chicken soup is part of three-generation family recipe tradition",
        "keywords": "tradition, family, heritage",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Soup",
        "u_target_name": "Family Recipe Tradition"
    },
    {
        "source_name": "Cooking Secrets",
        "target_name": "Family Recipe Tradition",
        "weight": 0.9,
        "description": "Cooking secrets are part of the family recipe tradition knowledge",
        "keywords": "secrets, knowledge, tradition",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Cooking Secrets",
        "u_target_name": "Family Recipe Tradition"
    },
    {
        "source_name": "Cooking Secrets",
        "target_name": "Cookie Baking Process",
        "weight": 0.7,
        "description": "Cooking secrets include tips for cookie baking like using room temperature butter",
        "keywords": "tips, butter, technique",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Cookie Baking Process",
        "u_target_name": "Cooking Secrets"
    },
    {
        "source_name": "Cooking Secrets",
        "target_name": "Chicken Broth",
        "weight": 0.7,
        "description": "Cooking secrets include using homemade broth for better soup",
        "keywords": "homemade, quality, technique",
        "source_id": "8de720dd-3311-475d-9c5b-bcabd28a3c1e",
        "filepath": "grandmas_recipes.md",
        "u_source_name": "Chicken Broth",
        "u_target_name": "Cooking Secrets"
    }
]