def icfts_explanation(model, feature_names, top_n=3):
    if hasattr(model, 'named_steps'):
        coef = model.named_steps['logisticregression'].coef_[0]
    else:
        coef = model.coef_[0]

    feature_contributions = list(zip(feature_names, coef))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_contributions[:top_n]

    explanation = "\n💡 ICFTS - Các đặc trưng ảnh hưởng mạnh nhất đến quyết định mô hình:"
    for name, val in top_features:
        direction = "tăng" if val > 0 else "giảm"
        explanation += f"\n- {name}: hệ số {round(val, 2)} → có xu hướng {direction} nguy cơ"

    return explanation
